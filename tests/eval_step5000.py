import os
import sys
import json
import time
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from model.loopsplit_frognano import LoopSplitFrogNano

MODEL_ID = "microsoft/FrogNano-4B-2609"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
CHECKPOINT_PATH = "/home/ubuntu/loopsplit_step5000.pt"

def evaluate_ce_and_generation():
    print(f"=== Starting Step 5000 Evaluation Suite on {DEVICE} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    base_model.eval()

    model = LoopSplitFrogNano(base_model, core_start=8, core_end=24, lora_rank=64).to(DEVICE)
    
    print(f"Loading checkpoint: {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.g2_mixer.data.copy_(ckpt["g2_mixer"])
    model.g2_mlp.data.copy_(ckpt["g2_mlp"])
    model.g3_mixer.data.copy_(ckpt["g3_mixer"])
    model.g3_mlp.data.copy_(ckpt["g3_mlp"])
    
    model_state = model.state_dict()
    for k, v in ckpt["state_dict"].items():
        if k in model_state:
            model_state[k].copy_(v)
    model.eval()
    print("Checkpoint loaded.\n")

    # 1. Held-Out TACO Validation CE Check (100 samples)
    print("Evaluating held-out TACO CE across loops (100 samples)...")
    dataset = load_dataset("parquet", data_files="hf://datasets/BAAI/TACO/ALL/train-*.parquet", split="train")
    
    val_samples = []
    for idx in range(len(dataset) - 1, len(dataset) - 200, -1):
        item = dataset[idx]
        solutions = item.get("solutions", [])
        if not solutions:
            continue
        code = solutions[0]
        prompt = item.get("question", "")
        chat_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{code}<|im_end|>"
        toks = tokenizer.encode(chat_text, max_length=512, truncation=True)
        if len(toks) >= 128:
            val_samples.append(torch.tensor(toks, dtype=torch.long))
        if len(val_samples) >= 100:
            break

    ce_l1_list, ce_l2_list, ce_l3_list = [], [], []
    with torch.no_grad():
        for s in val_samples:
            inp = s.unsqueeze(0).to(DEVICE)
            targets = inp[:, 1:].contiguous().view(-1)
            _, readouts = model(inp, num_loops=3, return_all_readouts=True)
            
            l1_logits = readouts["logits_loop_1"][:, :-1, :].contiguous().view(-1, readouts["logits_loop_1"].size(-1))
            l2_logits = readouts["logits_loop_2"][:, :-1, :].contiguous().view(-1, readouts["logits_loop_2"].size(-1))
            l3_logits = readouts["logits_loop_3"][:, :-1, :].contiguous().view(-1, readouts["logits_loop_3"].size(-1))
            
            ce_l1 = F.cross_entropy(l1_logits, targets).item()
            ce_l2 = F.cross_entropy(l2_logits, targets).item()
            ce_l3 = F.cross_entropy(l3_logits, targets).item()
            
            ce_l1_list.append(ce_l1)
            ce_l2_list.append(ce_l2)
            ce_l3_list.append(ce_l3)

    mean_ce_l1 = sum(ce_l1_list) / len(ce_l1_list)
    mean_ce_l2 = sum(ce_l2_list) / len(ce_l2_list)
    mean_ce_l3 = sum(ce_l3_list) / len(ce_l3_list)
    
    print(f"\n--- Held-Out TACO Validation CE (100 samples) ---")
    print(f"Loop 1 (Base path):       {mean_ce_l1:.4f} nats (PPL {math.exp(mean_ce_l1):.2f})")
    print(f"Loop 2 (Pass 2):          {mean_ce_l2:.4f} nats (PPL {math.exp(mean_ce_l2):.2f})")
    print(f"Loop 3 (Pass 3):          {mean_ce_l3:.4f} nats (PPL {math.exp(mean_ce_l3):.2f})")
    print(f"Delta (Loop 3 vs Loop 1): {mean_ce_l3 - mean_ce_l1:+.4f} nats")

    # 2. Qualitative Generations
    print("\n--- Generating Sample Completions ---")
    prompts = [
        "def count_distinct_characters(string: str) -> int:\n    \"\"\" Given a string, find how many distinct characters (case-insensitive) it consists of\n    >>> count_distinct_characters('xyzXYZ')\n    3\n    >>> count_distinct_characters('Jerry')\n    4\n    \"\"\"\n",
        "def find_zero(xs: list) -> float:\n    \"\"\" xs are coefficients of a polynomial. find_zero finds a point x with poly(x) = 0.\n    find_zero only takes list xs having even number of coefficients\n    and largest non zero coeff as it guarantees a solution.\n    >>> round(find_zero([1, 2]), 2) # f(x) = 1 + 2x\n    -0.5\n    \"\"\"\n",
        "def is_prime(n: int) -> bool:\n    \"\"\"Return true if a given number is prime, and false otherwise.\n    >>> is_prime(6)\n    False\n    >>> is_prime(101)\n    True\n    \"\"\"\n"
    ]
    
    for idx, p in enumerate(prompts, 1):
        print(f"\n[Test Prompt {idx}]:")
        print(p.strip())
        inp = tokenizer.encode(p, return_tensors="pt").to(DEVICE)
        curr = inp
        for _ in range(96):
            with torch.no_grad():
                logits = model(curr, num_loops=3)
                nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr = torch.cat([curr, nxt], dim=-1)
                if nxt.item() in [tokenizer.eos_token_id, 151643, 151645]:
                    break
        out = tokenizer.decode(curr[0][inp.shape[1]:], skip_special_tokens=True)
        print(f"[Generated (Loop 3)]:\n{out}")
        print("-" * 50)

if __name__ == "__main__":
    evaluate_ce_and_generation()
