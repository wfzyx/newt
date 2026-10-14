import os
import sys
import time
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from model.loopsplit_frognano import LoopSplitFrogNano
from train.regularized_loss import RegularizedLoopSplitLoss

MODEL_ID = "microsoft/FrogNano-4B-2609"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
PHASE_A_STEPS = 1500
MAX_LEN = 512
LR_LORA = 1e-4

def get_lr(step, total_steps, peak_lr, warmup_steps=150):
    if step < warmup_steps:
        return peak_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

def get_phase_a_gate(step, total_steps=1500):
    # Fixed linear ramp: 0.1 at step 1 -> 1.0 at step 1500
    return 0.1 + 0.9 * (step / total_steps)

def build_data_streams(tokenizer, target_count=2000):
    print("Preparing 50/30/20 data mixture for Phase A...", flush=True)
    samples = []
    
    # 1. 50% TACO (competitive programming)
    print("  Loading TACO stream...", flush=True)
    taco_ds = load_dataset("parquet", data_files="hf://datasets/BAAI/TACO/ALL/train-*.parquet", split="train")
    taco_count = int(target_count * 0.5)
    for item in taco_ds:
        solutions = item.get("solutions", [])
        if not solutions:
            continue
        text = f"# Problem\n{item.get('question', '').strip()}\n\n# Solution\n{solutions[0].strip()}"
        toks = tokenizer.encode(text, max_length=MAX_LEN, truncation=True)
        if len(toks) >= 96:
            samples.append(torch.tensor(toks, dtype=torch.long))
        if len(samples) >= taco_count:
            break

    # 2. 30% General Python (flytech/python-codes-25k)
    print("  Loading General Python stream...", flush=True)
    py_ds = load_dataset("flytech/python-codes-25k", split="train")
    py_count = int(target_count * 0.3)
    py_added = 0
    for item in py_ds:
        code_text = item.get("text", "")
        if not code_text:
            continue
        toks = tokenizer.encode(code_text, max_length=MAX_LEN, truncation=True)
        if len(toks) >= 64:
            samples.append(torch.tensor(toks, dtype=torch.long))
            py_added += 1
        if py_added >= py_count:
            break

    # 3. 20% Function-Completion Replay (google-research-datasets/mbpp)
    print("  Loading MBPP stream...", flush=True)
    mbpp_ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    mbpp_items = list(mbpp_ds)
    idx = 0
    while len(samples) < target_count and len(mbpp_items) > 0:
        item = mbpp_items[idx % len(mbpp_items)]
        prompt = item.get("prompt", "")
        code = item.get("code", "")
        text = f'"""\n{prompt}\n"""\n{code}'
        toks = tokenizer.encode(text, max_length=MAX_LEN, truncation=True)
        if len(toks) >= 48:
            samples.append(torch.tensor(toks, dtype=torch.long))
        idx += 1
        if idx >= target_count * 2:
            break

    import random
    random.Random(20261013).shuffle(samples)
    print(f"Total dataset prepared: {len(samples)} samples.", flush=True)
    return samples

def train_phase_a():
    print(f"=== Starting Phase A Training (Steps 1..{PHASE_A_STEPS}) on {DEVICE} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    
    # Initialize from Step 0 with state-routed LoRA
    model = LoopSplitFrogNano(base_model, core_start=8, core_end=24, lora_rank=64).to(DEVICE)
    model.train()
    
    # In Phase A: gates are fixed scalars (not trainable). Only LoRA parameters train.
    gate_params = [model.a2_mixer, model.a2_mlp, model.a3_mixer, model.a3_mlp]
    for gp in gate_params:
        gp.requires_grad = False
        
    lora_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(lora_params, lr=LR_LORA, weight_decay=0.01)
    # Advisor recommendation: lower KL anchor (0.01) in Phase A so LoRA can actively learn recurrence
    loss_fn = RegularizedLoopSplitLoss(w2=0.3, w3=1.0, beta_kl=0.01)
    
    train_tokens = build_data_streams(tokenizer, target_count=PHASE_A_STEPS + 200)
    
    start_time = time.time()
    for step in range(1, PHASE_A_STEPS + 1):
        cur_lr = get_lr(step, PHASE_A_STEPS, LR_LORA)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr
            
        cur_gate = get_phase_a_gate(step, PHASE_A_STEPS)
        input_ids = train_tokens[(step - 1) % len(train_tokens)].unsqueeze(0).to(DEVICE)
        
        optimizer.zero_grad()
        _, readouts = model(input_ids, num_loops=3, return_all_readouts=True, fixed_gate=cur_gate)
        
        loss_dict = loss_fn(readouts, input_ids)
        loss = loss_dict["loss"]
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        
        if step % 50 == 0 or step == 1:
            elapsed = time.time() - start_time
            sec_per_step = elapsed / max(1, step)
            rem_min = ((PHASE_A_STEPS - step) * sec_per_step) / 60.0
            
            # Check LoRA_B norm on Layer 0 to verify gradient updates
            b2_n = model.lora_layers[0].lora_B_2.norm().item()
            b3_n = model.lora_layers[0].lora_B_3.norm().item()
            
            print(f"Step {step:04d}/{PHASE_A_STEPS} | "
                  f"Gate: {cur_gate:.3f} | "
                  f"Loss: {loss.item():.4f} | "
                  f"CE Base: {loss_dict['ce_loop1_base']:.3f} | "
                  f"CE L2: {loss_dict['ce_loop2']:.3f} | "
                  f"CE L3: {loss_dict['ce_loop3']:.3f} | "
                  f"KL: {loss_dict['kl_anchor']:.4f} | "
                  f"B2: {b2_n:.4f} | B3: {b3_n:.4f} | "
                  f"ETA: {rem_min:.1f}m ({sec_per_step:.2f}s/it)", flush=True)

        if step % 500 == 0 or step == PHASE_A_STEPS:
            ckpt_path = f"/home/ubuntu/loopsplit_phaseA_step{step}.pt"
            torch.save({
                "step": step,
                "fixed_gate": cur_gate,
                "a2_mixer": model.a2_mixer,
                "a2_mlp": model.a2_mlp,
                "a3_mixer": model.a3_mixer,
                "a3_mlp": model.a3_mlp,
                "state_dict": {k: v for k, v in model.state_dict().items() if "lora" in k or "a2" in k or "a3" in k}
            }, ckpt_path)
            print(f">>> Checkpoint saved: {ckpt_path} <<<", flush=True)

    print(f"\nPhase A completed in {(time.time() - start_time) / 60:.1f} minutes.", flush=True)

if __name__ == "__main__":
    train_phase_a()
