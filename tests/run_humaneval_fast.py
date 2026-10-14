import os
import sys
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evalplus.data import get_human_eval_plus
from evalplus.evaluate import evaluate

from model.loopsplit_frognano import LoopSplitFrogNano

MODEL_ID = "microsoft/FrogNano-4B-2609"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
CHECKPOINT_PATH = "/home/ubuntu/loopsplit_step5000.pt"
OUTPUT_DIR = "/home/ubuntu/humaneval_eval_results"

def run_humaneval_eval(num_loops=3, max_tasks=50):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, f"loopsplit_humaneval_loop{num_loops}_n{max_tasks}.jsonl")
    
    print(f"=== Running HumanEval with LoopSplit (Loops={num_loops}, Tasks={max_tasks}) on {DEVICE} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    base_model.eval()

    model = LoopSplitFrogNano(base_model, core_start=8, core_end=24, lora_rank=64).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint {CHECKPOINT_PATH}...", flush=True)
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
    task_keys = list(problems.keys())[:max_tasks]
    print(f"Evaluating {len(task_keys)} tasks...", flush=True)
    
    results = []
    total_start = time.time()
    for idx, task_id in enumerate(task_keys, 1):
        task_start = time.time()
        prompt = problems[task_id]["prompt"]
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        curr_ids = input_ids
        
        # HumanEval functions typically complete in 30-80 tokens; cap at 128
        for _ in range(128):
            with torch.no_grad():
                logits = model(curr_ids, num_loops=num_loops)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr_ids = torch.cat([curr_ids, next_token], dim=-1)
                
                # Check for stop tokens or function definition end
                tok_val = next_token.item()
                if tok_val in [tokenizer.eos_token_id, 151643, 151645]:
                    break
                text_so_far = tokenizer.decode(curr_ids[0][input_ids.shape[1]:])
                if "\ndef " in text_so_far or "\nclass " in text_so_far or "\nif __name__" in text_so_far:
                    break
                    
        completion = tokenizer.decode(curr_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        results.append({
            "task_id": task_id,
            "completion": completion
        })
        
        dt = time.time() - task_start
        print(f"[{idx:02d}/{len(task_keys)}] {task_id} generated in {dt:.2f}s", flush=True)
            
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"\nGenerations saved to {out_file}.", flush=True)
    print("Scoring with EvalPlus...", flush=True)
    # Filter benchmark to subset for evalplus scoring
    evaluate(dataset="humaneval", samples=out_file)

if __name__ == "__main__":
    loops = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    run_humaneval_eval(num_loops=loops, max_tasks=tasks)
