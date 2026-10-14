import os
import sys
import time
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from model.loopsplit_frognano import LoopSplitFrogNano
from train.loss import LoopSplitLoss

MODEL_ID = "microsoft/FrogNano-4B-2609"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
TOTAL_STEPS = 5000
MAX_LEN = 512
RESUME_CKPT = "/home/ubuntu/loopsplit_pilot_step300.pt"

# Cosine schedule params
PEAK_LR_LORA = 2e-4
MIN_LR_LORA = 2e-5
PEAK_LR_GATES = 1e-2
MIN_LR_GATES = 1e-3

def get_lr(step, total_steps, peak_lr, min_lr, warmup_steps=100):
    if step < warmup_steps:
        return min_lr + (peak_lr - min_lr) * (step / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def train_5000():
    print(f"=== Starting 5,000 Step Recursive Core Training on {DEVICE} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    
    model = LoopSplitFrogNano(base_model, core_start=8, core_end=24, lora_rank=64).to(DEVICE)
    
    # Resume from Step 300 pilot checkpoint
    start_step = 1
    if os.path.exists(RESUME_CKPT):
        print(f"Loading weights from {RESUME_CKPT}...")
        ckpt = torch.load(RESUME_CKPT, map_location=DEVICE)
        model.g2_mixer.data.copy_(ckpt["g2_mixer"])
        model.g2_mlp.data.copy_(ckpt["g2_mlp"])
        model.g3_mixer.data.copy_(ckpt["g3_mixer"])
        model.g3_mlp.data.copy_(ckpt["g3_mlp"])
        
        model_state = model.state_dict()
        for k, v in ckpt["state_dict"].items():
            if k in model_state:
                model_state[k].copy_(v)
        start_step = 301
        print(f"Resuming from Step {start_step}!")

    model.train()
    
    gate_params = [model.g2_mixer, model.g2_mlp, model.g3_mixer, model.g3_mlp]
    lora_params = [p for p in model.parameters() if p.requires_grad and not any(p is gp for gp in gate_params)]
    
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": PEAK_LR_LORA, "weight_decay": 0.01},
        {"params": gate_params, "lr": PEAK_LR_GATES, "weight_decay": 0.0},
    ])
    
    loss_fn = LoopSplitLoss(w1=0.1, w2=0.3, w3=1.0, lambda_mono=1.0)
    
    print("Loading TACO dataset...")
    dataset = load_dataset("parquet", data_files="hf://datasets/BAAI/TACO/ALL/train-*.parquet", split="train")
    
    print(f"Streaming tokenized items...")
    train_tokens = []
    for item in dataset:
        solutions = item.get("solutions", [])
        if not solutions:
            continue
        code = solutions[0]
        prompt = item.get("question", "")
        chat_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{code}<|im_end|>"
        toks = tokenizer.encode(chat_text, max_length=MAX_LEN, truncation=True)
        if len(toks) >= 128:
            train_tokens.append(torch.tensor(toks, dtype=torch.long))
        if len(train_tokens) >= TOTAL_STEPS + 500:
            break
            
    print(f"Prepared {len(train_tokens)} samples. Beginning training...")
    
    start_time = time.time()
    for step in range(start_step, TOTAL_STEPS + 1):
        # Update learning rates via cosine decay
        cur_lr_lora = get_lr(step, TOTAL_STEPS, PEAK_LR_LORA, MIN_LR_LORA)
        cur_lr_gates = get_lr(step, TOTAL_STEPS, PEAK_LR_GATES, MIN_LR_GATES)
        optimizer.param_groups[0]["lr"] = cur_lr_lora
        optimizer.param_groups[1]["lr"] = cur_lr_gates
        
        input_ids = train_tokens[(step - 1) % len(train_tokens)].unsqueeze(0).to(DEVICE)
        
        optimizer.zero_grad()
        _, readouts = model(input_ids, num_loops=3, return_all_readouts=True)
        
        loss_dict = loss_fn(readouts, input_ids)
        loss = loss_dict["loss"]
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 50 == 0 or step == start_step:
            elapsed = time.time() - start_time
            steps_done = step - start_step + 1
            sec_per_step = elapsed / max(1, steps_done)
            remaining_sec = (TOTAL_STEPS - step) * sec_per_step
            
            g2_m_rms = model.g2_mixer.pow(2).mean().sqrt().item()
            g3_m_rms = model.g3_mixer.pow(2).mean().sqrt().item()
            
            print(f"Step {step:04d}/{TOTAL_STEPS} | "
                  f"Total: {loss.item():.4f} | "
                  f"CE L1: {loss_dict['ce_loop1']:.4f} | "
                  f"CE L2: {loss_dict['ce_loop2']:.4f} | "
                  f"CE L3: {loss_dict['ce_loop3']:.4f} | "
                  f"G2: {g2_m_rms:.3f} | G3: {g3_m_rms:.3f} | "
                  f"ETA: {remaining_sec / 60:.1f}m ({sec_per_step:.2f}s/it)")
                  
        if step % 1000 == 0 or step == TOTAL_STEPS:
            ckpt_path = f"/home/ubuntu/loopsplit_step{step}.pt"
            torch.save({
                "step": step,
                "g2_mixer": model.g2_mixer,
                "g2_mlp": model.g2_mlp,
                "g3_mixer": model.g3_mixer,
                "g3_mlp": model.g3_mlp,
                "state_dict": {k: v for k, v in model.state_dict().items() if "lora" in k or "g2" in k or "g3" in k}
            }, ckpt_path)
            print(f">>> Checkpoint saved: {ckpt_path} <<<")

    print(f"\nTraining completed in {(time.time() - start_time) / 60:.1f} minutes.")

if __name__ == "__main__":
    train_5000()
