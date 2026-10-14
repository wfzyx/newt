import os
import sys
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evalplus.data import get_human_eval_plus
from model.loopsplit_frognano import LoopSplitFrogNano

MODEL_ID = "microsoft/FrogNano-4B-2609"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
STOP_SEQS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#"]

def truncate_at_stop(text):
    first_idx = len(text)
    for s in STOP_SEQS:
        pos = text.find(s)
        if pos != -1 and pos < first_idx:
            first_idx = pos
    return text[:first_idx]

def run_probe(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found.")
        return

    print(f"\n=======================================================", flush=True)
    print(f"Testing Checkpoint: {ckpt_path}", flush=True)
    print(f"=======================================================", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    base_model.eval()

    model = LoopSplitFrogNano(base_model, core_start=8, core_end=24, lora_rank=64).to(DEVICE)
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.a2_mixer.data.copy_(ckpt["a2_mixer"])
    model.a2_mlp.data.copy_(ckpt["a2_mlp"])
    model.a3_mixer.data.copy_(ckpt["a3_mixer"])
    model.a3_mlp.data.copy_(ckpt["a3_mlp"])
    
    model_state = model.state_dict()
    for k, v in ckpt["state_dict"].items():
        if k in model_state:
            model_state[k].copy_(v)
    model.eval()

    g2_norm = torch.tanh(model.a2_mixer).pow(2).mean().sqrt().item()
    g3_norm = torch.tanh(model.a3_mixer).pow(2).mean().sqrt().item()
    print(f"Trained Gate Scales: g2={g2_norm:.4f}, g3={g3_norm:.4f}")

    # 1. Qualitative Prompt Inspect
    prompts = [
        ("HumanEval/0 (Numbers Close)", "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"),
        ("HumanEval/1 (Paren Groups)", "def separate_paren_groups(paren_string: str) -> list[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced, each group as a string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n")
    ]
    
    print("\n--- Qualitative Completions ---")
    for name, p in prompts:
        inp = tokenizer.encode(p, return_tensors="pt").to(DEVICE)
        curr = inp
        for _ in range(80):
            with torch.no_grad():
                logits = model(curr, num_loops=3)
                nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr = torch.cat([curr, nxt], dim=-1)
                if nxt.item() in [tokenizer.eos_token_id, 151643, 151645]:
                    break
                txt = tokenizer.decode(curr[0][inp.shape[1]:])
                if any(s in txt for s in STOP_SEQS):
                    break
        comp = truncate_at_stop(tokenizer.decode(curr[0][inp.shape[1]:], skip_special_tokens=True))
        print(f"[{name}]:\n{comp.strip()}\n")

    # 2. Fast 10-Problem Pass@1 Check
    problems = get_human_eval_plus()
    task_keys = list(problems.keys())[:10]
    passed = 0
    for tid in task_keys:
        p = problems[tid]["prompt"]
        entry = problems[tid]["entry_point"]
        test = problems[tid]["test"]
        inp = tokenizer.encode(p, return_tensors="pt").to(DEVICE)
        curr = inp
        for _ in range(96):
            with torch.no_grad():
                logits = model(curr, num_loops=3)
                nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr = torch.cat([curr, nxt], dim=-1)
                if nxt.item() in [tokenizer.eos_token_id, 151643, 151645]:
                    break
                txt = tokenizer.decode(curr[0][inp.shape[1]:])
                if any(s in txt for s in STOP_SEQS):
                    break
        comp = truncate_at_stop(tokenizer.decode(curr[0][inp.shape[1]:], skip_special_tokens=True))
        code = p + comp + "\n" + test + "\ncheck(" + entry + ")\n"
        try:
            exec(code, {})
            passed += 1
        except:
            pass

    print(f"Pass@1 on first 10 HumanEval tasks: {passed}/10 ({passed/10*100:.1f}%)")
    print(f"=======================================================\n", flush=True)

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/loopsplit_corrected_step1000.pt"
    run_probe(ckpt)
