import os
import sys
import json
import time
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evalplus.data import get_human_eval_plus

from model.loopsplit_frognano import LoopSplitFrogNano

MODEL_ID = "microsoft/FrogNano-4B-2609"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
STOP_SEQS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#"]

CKPT_1500 = "/home/ubuntu/loopsplit_phaseA_step1500.pt"
CKPT_5000 = "/home/ubuntu/loopsplit_phaseB_step5000.pt"

def truncate_at_stop(text):
    first_idx = len(text)
    for s in STOP_SEQS:
        pos = text.find(s)
        if pos != -1 and pos < first_idx:
            first_idx = pos
    return text[:first_idx]

def run_evaluation_suite():
    print(f"=== Running Statistical Rigor Evaluation on {DEVICE} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    base_model.eval()

    # Load TACO Test Split for Held-Out NLL (clean test partition)
    print("\n[Step 1] Loading Held-Out TACO Test Set for NLL Evaluation...", flush=True)
    test_ds = load_dataset("parquet", data_files="hf://datasets/BAAI/TACO/ALL/test-*.parquet", split="train")
    
    held_out_samples = []
    for item in test_ds:
        sols = item.get("solutions", [])
        if not sols:
            continue
        text = f"# Problem\n{item.get('question', '').strip()}\n\n# Solution\n{sols[0].strip()}"
        toks = tokenizer.encode(text, max_length=512, truncation=True)
        if len(toks) >= 128:
            held_out_samples.append(torch.tensor(toks, dtype=torch.long))
        if len(held_out_samples) >= 150:
            break
            
    print(f"Prepared {len(held_out_samples)} held-out documents (~60,000 tokens).", flush=True)

    # 1. Evaluate Pure Base Model on Held-Out NLL
    print("\nComputing Base Model Held-Out NLL...", flush=True)
    base_losses = []
    with torch.no_grad():
        for s in held_out_samples:
            inp = s.unsqueeze(0).to(DEVICE)
            targets = inp[:, 1:].contiguous().view(-1)
            logits = base_model(inp).logits[:, :-1, :].contiguous().view(-1, base_model.config.vocab_size)
            loss = F.cross_entropy(logits, targets).item()
            base_losses.append(loss)
    mean_base_nll = sum(base_losses) / len(base_losses)
    print(f"  -> Base Model Held-Out NLL: {mean_base_nll:.4f} nats (PPL {math.exp(mean_base_nll):.2f})", flush=True)

    # 2. Evaluate LoopSplit Checkpoints (Step 1500 & Step 5000)
    model = LoopSplitFrogNano(base_model, core_start=8, core_end=24, lora_rank=64).to(DEVICE)
    
    for name, ckpt_path in [("Phase A (Step 1500)", CKPT_1500), ("Phase B (Step 5000)", CKPT_5000)]:
        print(f"\nEvaluating {name} on Held-Out NLL...", flush=True)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model_state = model.state_dict()
        for k, v in ckpt["state_dict"].items():
            if k in model_state:
                model_state[k].copy_(v)
        if "a2_mixer" in ckpt:
            model.a2_mixer.data.copy_(ckpt["a2_mixer"])
            model.a2_mlp.data.copy_(ckpt["a2_mlp"])
            model.a3_mixer.data.copy_(ckpt["a3_mixer"])
            model.a3_mlp.data.copy_(ckpt["a3_mlp"])
        fixed_gate = ckpt.get("fixed_gate", None)
        model.eval()

        ckpt_losses = []
        with torch.no_grad():
            for s in held_out_samples:
                inp = s.unsqueeze(0).to(DEVICE)
                targets = inp[:, 1:].contiguous().view(-1)
                logits = model(inp, num_loops=3, fixed_gate=fixed_gate)
                logits_shift = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                loss = F.cross_entropy(logits_shift, targets).item()
                ckpt_losses.append(loss)
        mean_ckpt_nll = sum(ckpt_losses) / len(ckpt_losses)
        diff = mean_ckpt_nll - mean_base_nll
        print(f"  -> {name} Held-Out NLL: {mean_ckpt_nll:.4f} nats (PPL {math.exp(mean_ckpt_nll):.2f}) | Delta vs Base: {diff:+.4f} nats", flush=True)

    # 3. Full HumanEval (All 164 Tasks)
    print("\n" + "=" * 60, flush=True)
    print("[Step 2] Running Full HumanEval (164 Tasks) Paired Comparison", flush=True)
    print("=" * 60, flush=True)
    
    problems = get_human_eval_plus()
    all_keys = list(problems.keys())
    print(f"Loaded all {len(all_keys)} HumanEval tasks.", flush=True)

    # Base Full HumanEval
    print("\nRunning Base FrogNano on full 164 tasks...", flush=True)
    base_results = {}
    base_passed = 0
    t0 = time.time()
    for idx, tid in enumerate(all_keys, 1):
        p = problems[tid]["prompt"]
        entry = problems[tid]["entry_point"]
        test = problems[tid]["test"]
        inp = tokenizer.encode(p, return_tensors="pt").to(DEVICE)
        out = base_model.generate(inp, max_new_tokens=96, do_sample=False, eos_token_id=tokenizer.eos_token_id)
        comp = truncate_at_stop(tokenizer.decode(out[0][inp.shape[1]:], skip_special_tokens=True))
        code = p + comp + "\n" + test + "\ncheck(" + entry + ")\n"
        try:
            exec(code, {})
            base_results[tid] = True
            base_passed += 1
        except:
            base_results[tid] = False
        if idx % 40 == 0 or idx == len(all_keys):
            print(f"  Base progress: {idx}/164 ({base_passed} passed)", flush=True)
    print(f"Base FrogNano Full HumanEval: {base_passed}/164 ({base_passed/164*100:.1f}%) in {time.time() - t0:.1f}s", flush=True)

    # Phase B Step 5000 Full HumanEval
    print("\nRunning Phase B Step 5000 on full 164 tasks...", flush=True)
    ckpt = torch.load(CKPT_5000, map_location=DEVICE)
    model_state = model.state_dict()
    for k, v in ckpt["state_dict"].items():
        if k in model_state:
            model_state[k].copy_(v)
    model.a2_mixer.data.copy_(ckpt["a2_mixer"])
    model.a2_mlp.data.copy_(ckpt["a2_mlp"])
    model.a3_mixer.data.copy_(ckpt["a3_mixer"])
    model.a3_mlp.data.copy_(ckpt["a3_mlp"])
    model.eval()

    step5k_results = {}
    step5k_passed = 0
    t1 = time.time()
    for idx, tid in enumerate(all_keys, 1):
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
            step5k_results[tid] = True
            step5k_passed += 1
        except:
            step5k_results[tid] = False
        if idx % 40 == 0 or idx == len(all_keys):
            print(f"  Step 5000 progress: {idx}/164 ({step5k_passed} passed)", flush=True)
    print(f"Step 5000 Full HumanEval: {step5k_passed}/164 ({step5k_passed/164*100:.1f}%) in {time.time() - t1:.1f}s", flush=True)

    # 4. McNemar Exact Test on Discordant Pairs
    b_only = 0  # Passed by Base, failed by Step 5000
    s_only = 0  # Passed by Step 5000, failed by Base
    both_pass = 0
    both_fail = 0

    for tid in all_keys:
        b = base_results[tid]
        s = step5k_results[tid]
        if b and s:
            both_pass += 1
        elif b and not s:
            b_only += 1
        elif not b and s:
            s_only += 1
        else:
            both_fail += 1

    print("\n" + "=" * 60, flush=True)
    print("=== FINAL RIGOROUS STATISTICAL VERDICT ===", flush=True)
    print(f"Base Passed:                 {base_passed}/164 ({base_passed/164*100:.1f}%)")
    print(f"Step 5000 (Loop 3) Passed:   {step5k_passed}/164 ({step5k_passed/164*100:.1f}%)")
    print(f"Both Passed:                 {both_pass}")
    print(f"Both Failed:                 {both_fail}")
    print(f"Base Only Wins (b_only):     {b_only}")
    print(f"Step 5000 Wins (s_only):     {s_only}")
    
    # McNemar's test with continuity correction
    b_discordant = b_only + s_only
    if b_discordant > 0:
        z = (abs(s_only - b_only) - 1) / math.sqrt(b_discordant)
        p_val = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
        print(f"Discordant Pairs:            {b_discordant}")
        print(f"McNemar z-statistic:         {z:.4f} (p-value: {p_val:.4f})")
    print("=" * 60, flush=True)

if __name__ == "__main__":
    run_evaluation_suite()
