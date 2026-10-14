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

CHECKPOINTS = [
    ("Step 5,000 (Phase B)", "/home/ubuntu/loopsplit_phaseB_step5000.pt"),
    ("Step 6,000 (Phase C)", "/home/ubuntu/loopsplit_phaseC_step6000.pt"),
    ("Step 7,000 (Phase C)", "/home/ubuntu/loopsplit_phaseC_step7000.pt"),
    ("Step 8,000 (Phase C)", "/home/ubuntu/loopsplit_phaseC_step8000.pt"),
    ("Step 9,000 (Phase C)", "/home/ubuntu/loopsplit_phaseC_step9000.pt"),
    ("Step 10,000 (Phase C)", "/home/ubuntu/loopsplit_phaseC_step10000.pt"),
]

def truncate_at_stop(text):
    first_idx = len(text)
    for s in STOP_SEQS:
        pos = text.find(s)
        if pos != -1 and pos < first_idx:
            first_idx = pos
    return text[:first_idx]

def load_checkpoint_into_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = model.state_dict()
    ckpt_sd = ckpt["state_dict"]
    for k, v in ckpt_sd.items():
        if k in state:
            state[k].copy_(v)
    if "a2_mixer" in ckpt:
        model.a2_mixer.data.copy_(ckpt["a2_mixer"])
        model.a2_mlp.data.copy_(ckpt["a2_mlp"])
        model.a3_mixer.data.copy_(ckpt["a3_mixer"])
        model.a3_mlp.data.copy_(ckpt["a3_mlp"])
    model.eval()
    return ckpt

def evaluate_nll(forward_fn, samples):
    losses = []
    with torch.no_grad():
        for s in samples:
            inp = s.unsqueeze(0).to(DEVICE)
            targets = inp[:, 1:].contiguous().view(-1)
            logits = forward_fn(inp)
            logits_shift = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            loss = F.cross_entropy(logits_shift, targets).item()
            losses.append(loss)
    return sum(losses) / len(losses)

def generate_greedy(forward_fn, tokenizer, prompt_ids, max_new_tokens=96):
    curr = prompt_ids
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = forward_fn(curr)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr = torch.cat([curr, nxt], dim=-1)
            if nxt.item() in [tokenizer.eos_token_id, 151643, 151645]:
                break
            txt = tokenizer.decode(curr[0][prompt_ids.shape[1]:])
            if any(s in txt for s in STOP_SEQS):
                break
    comp = truncate_at_stop(tokenizer.decode(curr[0][prompt_ids.shape[1]:], skip_special_tokens=True))
    return comp

def main():
    print(f"=== CHECKPOINT TREND BENCHMARK SUITE (Base -> 5k -> 6k -> 7k -> 8k -> 9k -> 10k) ===", flush=True)
    print(f"Device: {DEVICE} | Precision: {DTYPE}\n", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    base_model.eval()

    # Step 1: Held-out TACO Test Set (150 docs ~60k tokens)
    print("[1/3] Loading Held-Out TACO Test Set...", flush=True)
    test_ds = load_dataset("parquet", data_files="hf://datasets/BAAI/TACO/ALL/test-*.parquet", split="train")
    held_out_samples = []
    for item in test_ds:
        sols = item.get("solutions", [])
        if not sols:
            continue
        if isinstance(sols, str):
            try:
                sols = json.loads(sols)
            except:
                continue
        if not sols or len(sols[0].strip()) < 40:
            continue
        text = f"# Problem\n{item.get('question', '').strip()}\n\n# Solution\n{sols[0].strip()}"
        toks = tokenizer.encode(text, max_length=512, truncation=True)
        if len(toks) >= 128:
            held_out_samples.append(torch.tensor(toks, dtype=torch.long))
        if len(held_out_samples) >= 150:
            break
    print(f"  Loaded {len(held_out_samples)} held-out TACO test documents.\n", flush=True)

    # Step 2: HumanEval subset (first 40 tasks for fast high-resolution trend tracking)
    print("[2/3] Loading HumanEval Tasks...", flush=True)
    problems = get_human_eval_plus()
    all_keys = list(problems.keys())[:40]
    print(f"  Selected first {len(all_keys)} HumanEval tasks for standardized comparative evaluation.\n", flush=True)

    # Wrap model with LoopSplit
    loopsplit_model = LoopSplitFrogNano(base_model, core_start=8, core_end=24, lora_rank=64).to(DEVICE)
    loopsplit_model.eval()

    summary_records = []

    # --- BASE EVALUATION ---
    print("=" * 60, flush=True)
    print("EVALUATING: Pure Base FrogNano-4B", flush=True)
    print("=" * 60, flush=True)
    base_nll = evaluate_nll(lambda x: base_model(x).logits, held_out_samples)
    print(f"  Base Held-out NLL: {base_nll:.4f} nats (PPL: {math.exp(base_nll):.2f})", flush=True)

    base_passes = 0
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
            base_passes += 1
        except:
            pass
    print(f"  Base HumanEval-40: {base_passes}/{len(all_keys)} ({base_passes/len(all_keys)*100:.1f}%) in {time.time()-t0:.1f}s\n", flush=True)
    
    summary_records.append({
        "name": "Base FrogNano-4B",
        "nll": base_nll,
        "delta_nll": 0.0,
        "ppl": math.exp(base_nll),
        "he40_passed": base_passes,
        "he40_total": len(all_keys),
        "he40_pct": base_passes / len(all_keys) * 100.0,
        "g2": 0.0,
        "g3": 0.0
    })

    # --- CHECKPOINTS EVALUATION ---
    for name, ckpt_path in CHECKPOINTS:
        print("=" * 60, flush=True)
        print(f"EVALUATING: {name}", flush=True)
        print("=" * 60, flush=True)
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: Checkpoint {ckpt_path} not found. Skipping.", flush=True)
            continue
            
        load_checkpoint_into_model(loopsplit_model, ckpt_path)
        g2 = torch.tanh(loopsplit_model.a2_mixer).mean().item()
        g3 = torch.tanh(loopsplit_model.a3_mixer).mean().item()

        ckpt_nll = evaluate_nll(lambda x: loopsplit_model(x, num_loops=3), held_out_samples)
        delta_nll = ckpt_nll - base_nll
        print(f"  Held-out NLL: {ckpt_nll:.4f} nats (PPL: {math.exp(ckpt_nll):.2f}) | Delta vs Base: {delta_nll:+.4f} nats", flush=True)

        ckpt_passes = 0
        t_start = time.time()
        for idx, tid in enumerate(all_keys, 1):
            p = problems[tid]["prompt"]
            entry = problems[tid]["entry_point"]
            test = problems[tid]["test"]
            inp = tokenizer.encode(p, return_tensors="pt").to(DEVICE)
            comp = generate_greedy(lambda x: loopsplit_model(x, num_loops=3), tokenizer, inp, max_new_tokens=96)
            code = p + comp + "\n" + test + "\ncheck(" + entry + ")\n"
            try:
                exec(code, {})
                ckpt_passes += 1
            except:
                pass
            if idx % 20 == 0:
                print(f"    Progress: {idx}/{len(all_keys)} ({ckpt_passes} passed)", flush=True)
        print(f"  HumanEval-40: {ckpt_passes}/{len(all_keys)} ({ckpt_passes/len(all_keys)*100:.1f}%) in {time.time()-t_start:.1f}s\n", flush=True)

        summary_records.append({
            "name": name,
            "nll": ckpt_nll,
            "delta_nll": delta_nll,
            "ppl": math.exp(ckpt_nll),
            "he40_passed": ckpt_passes,
            "he40_total": len(all_keys),
            "he40_pct": ckpt_passes / len(all_keys) * 100.0,
            "g2": g2,
            "g3": g3
        })

    # --- PRINT COMPARATIVE TREND TABLE ---
    print("\n" + "=" * 80, flush=True)
    print("=== SUMMARY OF CHECKPOINT BENCHMARK TRENDS ===", flush=True)
    print(f"{'Model / Checkpoint':<25} | {'NLL (nats)':<10} | {'Delta':<10} | {'PPL':<6} | {'HE-40 Pass':<12} | {'g2/g3':<10}", flush=True)
    print("-" * 80, flush=True)
    for rec in summary_records:
        g_str = f"{rec['g2']:.3f}/{rec['g3']:.3f}" if rec['g2'] > 0 else "0.000/0.000"
        print(f"{rec['name']:<25} | {rec['nll']:<10.4f} | {rec['delta_nll']:<+10.4f} | {rec['ppl']:<6.2f} | {rec['he40_passed']:>2}/{rec['he40_total']} ({rec['he40_pct']:>4.1f}%) | {g_str:<10}", flush=True)
    print("=" * 80, flush=True)

    with open("/home/ubuntu/checkpoint_trends_summary.json", "w") as f:
        json.dump(summary_records, f, indent=2)
    print("\nSaved full summary to /home/ubuntu/checkpoint_trends_summary.json", flush=True)

if __name__ == "__main__":
    main()
