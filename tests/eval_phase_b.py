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
CHECKPOINT_PATH = "/home/ubuntu/loopsplit_phaseB_step5000.pt"
STOP_SEQS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#"]

def truncate_at_stop(text):
    first_idx = len(text)
    for s in STOP_SEQS:
        pos = text.find(s)
        if pos != -1 and pos < first_idx:
            first_idx = pos
    return text[:first_idx]

def eval_phase_b():
    print(f"=== Evaluating Phase B Checkpoint (Step 5000) on {DEVICE} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    base_model.eval()

    model = LoopSplitFrogNano(base_model, core_start=8, core_end=24, lora_rank=64).to(DEVICE)
    
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.a2_mixer.data.copy_(ckpt["a2_mixer"])
    model.a2_mlp.data.copy_(ckpt["a2_mlp"])
    model.a3_mixer.data.copy_(ckpt["a3_mixer"])
    model.a3_mlp.data.copy_(ckpt["a3_mlp"])
    
    model_state = model.state_dict()
    for k, v in ckpt["state_dict"].items():
        if k in model_state:
            model_state[k].copy_(v)
    model.eval()

    # Verify gates & LoRA weight norms
    g2_scale = torch.tanh(model.a2_mixer).mean().item()
    g3_scale = torch.tanh(model.a3_mixer).mean().item()
    b2_norm = model.lora_layers[0].lora_B_2.norm().item()
    b3_norm = model.lora_layers[0].lora_B_3.norm().item()
    print(f"Trained Gates: g2={g2_scale:.4f}, g3={g3_scale:.4f}", flush=True)
    print(f"Layer 0 LoRA B norms: B2={b2_norm:.4f}, B3={b3_norm:.4f}", flush=True)

    # 1. Qualitative Prompt Inspection
    prompts = [
        ("HumanEval/0 (Numbers Close)", "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"),
        ("HumanEval/1 (Paren Groups)", "def separate_paren_groups(paren_string: str) -> list[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced, each group as a string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n")
    ]
    
    print("\n--- Qualitative Completions (Loop 3, learned gates) ---", flush=True)
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
        print(f"[{name}]:\n{comp.strip()}\n", flush=True)

    # 2. 30-Problem Pass@1 Benchmark Check
    problems = get_human_eval_plus()
    task_keys = list(problems.keys())[:30]
    passed = 0
    t0 = time.time()
    for idx, tid in enumerate(task_keys, 1):
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
        if idx % 10 == 0:
            print(f"  Progress: {idx}/30 (running passed: {passed})", flush=True)

    acc = (passed / 30) * 100
    print(f"\nPhase B Step 5000 Pass@1 (30 tasks): {passed}/30 ({acc:.1f}%) in {time.time() - t0:.1f}s", flush=True)

if __name__ == "__main__":
    eval_phase_b()
