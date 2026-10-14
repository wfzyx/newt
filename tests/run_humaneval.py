import os
import sys
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evalplus.data import get_human_eval_plus
from evalplus.evaluate import evaluate

from model.loopsplit_frognano import LoopSplitFrogNano

MODEL_ID = "microsoft/FrogNano-4B-2609"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
CHECKPOINT_PATH = "/home/ubuntu/loopsplit_step5000.pt"
OUTPUT_DIR = "/home/ubuntu/humaneval_eval_results"

def run_humaneval(num_loops=3):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, f"loopsplit_humaneval_loop{num_loops}.jsonl")
    
    print(f"=== Running HumanEval with LoopSplit (Loops={num_loops}) on {DEVICE} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    base_model.eval()

    model = LoopSplitFrogNano(base_model, core_start=8, core_end=24, lora_rank=64).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint {CHECKPOINT_PATH}...")
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

    problems = get_human_eval_plus()
    print(f"Loaded {len(problems)} HumanEval tasks.")
    
    results = []
    start_time = time.time()
    for idx, (task_id, task_data) in enumerate(problems.items(), 1):
        prompt = task_data["prompt"]
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        curr_ids = input_ids
        
        # Greedy autoregressive decode up to 256 tokens
        for _ in range(256):
            with torch.no_grad():
                logits = model(curr_ids, num_loops=num_loops)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr_ids = torch.cat([curr_ids, next_token], dim=-1)
                if next_token.item() in [tokenizer.eos_token_id, 151643, 151645]:
                    break
                    
        completion = tokenizer.decode(curr_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        results.append({
            "task_id": task_id,
            "completion": completion
        })
        
        if idx % 20 == 0 or idx == len(problems):
            elapsed = time.time() - start_time
            print(f"Completed {idx:03d}/{len(problems)} tasks ({elapsed:.1f}s)...")
            
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"\nGenerations saved to {out_file}.")
    print("Scoring with EvalPlus...")
    evaluate(dataset="humaneval", samples=out_file)

if __name__ == "__main__":
    loops = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run_humaneval(num_loops=loops)
