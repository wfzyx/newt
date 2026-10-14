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
MAX_STEPS = 300
MAX_LEN = 512
LR_LORA = 2e-4
LR_GATES = 1e-2

def train_pilot():
    print(f"=== Starting Gate 1 Pilot ({MAX_STEPS} steps, MAX_LEN={MAX_LEN}) on {DEVICE} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    
    # Wrap with LoopSplit (Core 8..23, 3 loops, rank 64)
    model = LoopSplitFrogNano(base_model, core_start=8, core_end=24, lora_rank=64).to(DEVICE)
    model.train()
    
    # Prepare parameters: separate groups for LoRA and ReZero gates
    gate_params = [model.g2_mixer, model.g2_mlp, model.g3_mixer, model.g3_mlp]
    lora_params = [p for p in model.parameters() if p.requires_grad and not any(p is gp for gp in gate_params)]
    
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LR_LORA, "weight_decay": 0.01},
        {"params": gate_params, "lr": LR_GATES, "weight_decay": 0.0},
    ])
    
    loss_fn = LoopSplitLoss(w1=0.1, w2=0.3, w3=1.0, lambda_mono=1.0)
    
    # Load TACO dataset
    print("Loading TACO dataset...")
    dataset = load_dataset("parquet", data_files="hf://datasets/BAAI/TACO/ALL/train-*.parquet", split="train")
    
    print(f"Dataset loaded ({len(dataset)} examples). Preparing pilot queue...")
    train_tokens = []
    for item in dataset:
        solutions = item.get("solutions", [])
        if not solutions:
            continue
        code = solutions[0]
        prompt = item.get("question", "")
        # Apply standard chat formatting
        chat_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{code}<|im_end|>"
        toks = tokenizer.encode(chat_text, max_length=MAX_LEN, truncation=True)
        if len(toks) >= 128:
            train_tokens.append(torch.tensor(toks, dtype=torch.long))
        if len(train_tokens) >= MAX_STEPS + 50:
            break
            
    print(f"Loaded {len(train_tokens)} samples for training.")
    
    # Training Loop
    start_time = time.time()
    for step in range(1, MAX_STEPS + 1):
        input_ids = train_tokens[step - 1].unsqueeze(0).to(DEVICE)
        
        optimizer.zero_grad()
        _, readouts = model(input_ids, num_loops=3, return_all_readouts=True)
        
        loss_dict = loss_fn(readouts, input_ids)
        loss = loss_dict["loss"]
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 25 == 0 or step == 1:
            elapsed = time.time() - start_time
            g2_m_rms = model.g2_mixer.pow(2).mean().sqrt().item()
            g2_mlp_rms = model.g2_mlp.pow(2).mean().sqrt().item()
            g3_m_rms = model.g3_mixer.pow(2).mean().sqrt().item()
            g3_mlp_rms = model.g3_mlp.pow(2).mean().sqrt().item()
            
            print(f"Step {step:03d}/{MAX_STEPS} | "
                  f"Total Loss: {loss.item():.4f} | "
                  f"CE L1: {loss_dict['ce_loop1']:.4f} | "
                  f"CE L2: {loss_dict['ce_loop2']:.4f} | "
                  f"CE L3: {loss_dict['ce_loop3']:.4f} | "
                  f"G2 RMS: {g2_m_rms:.4f} | "
                  f"G3 RMS: {g3_m_rms:.4f} | "
                  f"{elapsed:.1f}s")
                  
    print(f"\nGate 1 Pilot completed in {time.time() - start_time:.1f}s.")
    torch.save({
        "g2_mixer": model.g2_mixer,
        "g2_mlp": model.g2_mlp,
        "g3_mixer": model.g3_mixer,
        "g3_mlp": model.g3_mlp,
        "state_dict": {k: v for k, v in model.state_dict().items() if "lora" in k or "g2" in k or "g3" in k}
    }, "/home/ubuntu/loopsplit_pilot_step300.pt")
    print("Checkpoint saved to /home/ubuntu/loopsplit_pilot_step300.pt")

if __name__ == "__main__":
    train_pilot()
