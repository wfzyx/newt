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
START_STEP = 1501
TOTAL_STEPS = 5000
MAX_LEN = 512
LR_LORA = 1e-4
LR_GATES = 1e-4
RESUME_CKPT = "/home/ubuntu/loopsplit_phaseA_step1500.pt"

def get_lr(step, total_steps, peak_lr, warmup_steps=100):
    # Cosine decay over remaining steps
    step_offset = step - START_STEP
    total_phase_b = total_steps - START_STEP
    if step_offset < warmup_steps:
        return peak_lr * (step_offset / warmup_steps)
    progress = (step_offset - warmup_steps) / max(1, total_phase_b - warmup_steps)
    return peak_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

def build_data_streams(tokenizer, target_count=4000):
    print("Preparing 50/30/20 data mixture for Phase B...", flush=True)
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
    random.Random(20261014).shuffle(samples)
    print(f"Total dataset prepared: {len(samples)} samples.", flush=True)
    return samples

def train_phase_b():
    print(f"=== Starting Phase B Training (Steps {START_STEP}..{TOTAL_STEPS}) on {DEVICE} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    
    model = LoopSplitFrogNano(base_model, core_start=8, core_end=24, lora_rank=64).to(DEVICE)
    
    # Load Phase A Step 1500 checkpoint
    print(f"Loading Phase A checkpoint: {RESUME_CKPT}...", flush=True)
    ckpt = torch.load(RESUME_CKPT, map_location=DEVICE)
    model_state = model.state_dict()
    for k, v in ckpt["state_dict"].items():
        if k in model_state:
            model_state[k].copy_(v)
            
    # Initialize gate logits continuous with Phase A (tanh(2.0) = 0.964)
    with torch.no_grad():
        model.a2_mixer.data.fill_(2.0)
        model.a2_mlp.data.fill_(2.0)
        model.a3_mixer.data.fill_(2.0)
        model.a3_mlp.data.fill_(2.0)

    model.train()
    
    # Unfreeze gates: trainable with weight decay = 0.0
    gate_params = [model.a2_mixer, model.a2_mlp, model.a3_mixer, model.a3_mlp]
    for gp in gate_params:
        gp.requires_grad = True
        
    lora_params = [p for p in model.parameters() if p.requires_grad and not any(p is gp for gp in gate_params)]
    
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LR_LORA, "weight_decay": 0.01},
        {"params": gate_params, "lr": LR_GATES, "weight_decay": 0.0},
    ])
    
    # Phase B loss: restore KL anchor to 0.05
    loss_fn = RegularizedLoopSplitLoss(w2=0.3, w3=1.0, beta_kl=0.05)
    train_tokens = build_data_streams(tokenizer, target_count=TOTAL_STEPS - START_STEP + 300)
    
    start_time = time.time()
    for step in range(START_STEP, TOTAL_STEPS + 1):
        cur_lr_lora = get_lr(step, TOTAL_STEPS, LR_LORA)
        cur_lr_gates = get_lr(step, TOTAL_STEPS, LR_GATES)
        optimizer.param_groups[0]["lr"] = cur_lr_lora
        optimizer.param_groups[1]["lr"] = cur_lr_gates
        
        input_ids = train_tokens[(step - START_STEP) % len(train_tokens)].unsqueeze(0).to(DEVICE)
        
        optimizer.zero_grad()
        # In Phase B: fixed_gate=None, model uses learned bounded tanh(a_k) gates
        _, readouts = model(input_ids, num_loops=3, return_all_readouts=True, fixed_gate=None)
        
        loss_dict = loss_fn(readouts, input_ids)
        loss = loss_dict["loss"]
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
        optimizer.step()
        
        # Clip gate logits to [-3.0, 3.0] to prevent tanh saturation
        with torch.no_grad():
            for gp in gate_params:
                gp.clamp_(-3.0, 3.0)
                
        if step % 50 == 0 or step == START_STEP:
            elapsed = time.time() - start_time
            steps_done = step - START_STEP + 1
            sec_per_step = elapsed / max(1, steps_done)
            rem_min = ((TOTAL_STEPS - step) * sec_per_step) / 60.0
            
            g2_m = torch.tanh(model.a2_mixer).mean().item()
            g3_m = torch.tanh(model.a3_mixer).mean().item()
            b2_n = model.lora_layers[0].lora_B_2.norm().item()
            b3_n = model.lora_layers[0].lora_B_3.norm().item()
            
            print(f"Step {step:04d}/{TOTAL_STEPS} | "
                  f"Loss: {loss.item():.4f} | "
                  f"CE Base: {loss_dict['ce_loop1_base']:.3f} | "
                  f"CE L2: {loss_dict['ce_loop2']:.3f} | "
                  f"CE L3: {loss_dict['ce_loop3']:.3f} | "
                  f"KL: {loss_dict['kl_anchor']:.4f} | "
                  f"g2: {g2_m:.3f} | g3: {g3_m:.3f} | "
                  f"B2: {b2_n:.3f} | B3: {b3_n:.3f} | "
                  f"ETA: {rem_min:.1f}m ({sec_per_step:.2f}s/it)", flush=True)

        if step % 1000 == 0 or step == TOTAL_STEPS:
            ckpt_path = f"/home/ubuntu/loopsplit_phaseB_step{step}.pt"
            torch.save({
                "step": step,
                "a2_mixer": model.a2_mixer,
                "a2_mlp": model.a2_mlp,
                "a3_mixer": model.a3_mixer,
                "a3_mlp": model.a3_mlp,
                "state_dict": {k: v for k, v in model.state_dict().items() if "lora" in k or "a2" in k or "a3" in k}
            }, ckpt_path)
            print(f">>> Checkpoint saved: {ckpt_path} <<<", flush=True)

    print(f"\nPhase B completed in {(time.time() - start_time) / 60:.1f} minutes.", flush=True)

if __name__ == "__main__":
    train_phase_b()
