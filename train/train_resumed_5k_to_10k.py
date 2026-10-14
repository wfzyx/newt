import os
import sys
import time
import math
import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evalplus.data import get_human_eval_plus

from model.loopsplit_frognano import LoopSplitFrogNano
from train.hinge_loss import HingeTrustRegionLoopSplitLoss

MODEL_ID = "microsoft/FrogNano-4B-2609"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

START_STEP = 5001
TOTAL_STEPS = 10000
MAX_LEN = 512

PEAK_LR_LORA = 5e-5    # 50% of Phase B peak
MIN_LR_LORA = 1e-5     # 10% of Phase B peak
PEAK_LR_GATES = 5e-6   # 0.1x LoRA LR
MIN_LR_GATES = 1e-6
WARMUP_STEPS = 200

RESUME_CKPT = "/home/ubuntu/loopsplit_phaseB_step5000.pt"

def get_cosine_lr(step, start_step, total_steps, peak_lr, min_lr, warmup_steps=200):
    offset = step - start_step
    total_run = total_steps - start_step
    if offset < warmup_steps:
        return min_lr + (peak_lr - min_lr) * (offset / warmup_steps)
    progress = (offset - warmup_steps) / max(1, total_run - warmup_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def build_13gram_set(texts):
    ngrams = set()
    for t in texts:
        tokens = t.split()
        for i in range(len(tokens) - 12):
            ngrams.add(" ".join(tokens[i:i+13]))
    return ngrams

def build_upgraded_dataset(tokenizer, target_count=5500):
    print("=== Building Upgraded Decontaminated Dataset (No MBPP Repetition) ===", flush=True)
    
    print("  Indexing HumanEval 13-grams for contamination audit...", flush=True)
    he_problems = get_human_eval_plus()
    test_texts = [p["prompt"] + p.get("canonical_solution", "") for p in he_problems.values()]
    decontam_ngrams = build_13gram_set(test_texts)
    print(f"  Indexed {len(decontam_ngrams)} 13-grams from benchmark test sets.", flush=True)

    def is_contaminated(text):
        tokens = text.split()
        for i in range(len(tokens) - 12):
            if " ".join(tokens[i:i+13]) in decontam_ngrams:
                return True
        return False

    samples = []
    stream_counts = {"taco": 0, "contests": 0, "mbpp": 0, "flytech": 0, "func_impl": 0}
    dropped_contamination = 0
    first_taco_preview = None

    # 1. TACO stream
    print("  [1/5] Loading TACO stream...", flush=True)
    taco_ds = load_dataset("parquet", data_files="hf://datasets/BAAI/TACO/ALL/train-*.parquet", split="train")
    taco_iter = iter(taco_ds)
    taco_target = int(target_count * 0.45)
    
    for item in taco_iter:
        sols_raw = item.get("solutions", "[]")
        solutions = json.loads(sols_raw) if isinstance(sols_raw, str) else sols_raw
        if not solutions or not isinstance(solutions, list):
            continue
        code = solutions[0].strip()
        if len(code) < 40:
            continue
        problem = item.get("question", "").strip()
        text = f"# Problem\n{problem}\n\n# Solution\n{code}"
        if is_contaminated(text):
            dropped_contamination += 1
            continue
        toks = tokenizer.encode(text, max_length=MAX_LEN, truncation=True)
        if len(toks) >= 96:
            samples.append(torch.tensor(toks, dtype=torch.long))
            stream_counts["taco"] += 1
            if first_taco_preview is None:
                first_taco_preview = text[-250:]
        if stream_counts["taco"] >= taco_target:
            break

    # 2. Hard Algorithmic Contest Data (deepmind/code_contests)
    print("  [2/5] Loading Contest stream (code_contests)...", flush=True)
    contest_target = int(target_count * 0.20)
    try:
        contest_ds = load_dataset("deepmind/code_contests", split="train", streaming=True)
        for item in contest_ds:
            solutions = item.get("solutions", {}).get("solution", [])
            languages = item.get("solutions", {}).get("language", [])
            py_sol = None
            for s, lang in zip(solutions, languages):
                if lang == 3 and s.strip():
                    py_sol = s.strip()
                    break
            if not py_sol or len(py_sol) < 40:
                continue
            desc = item.get("description", "").strip()
            text = f"# Contest Problem\n{desc}\n\n# Solution\n{py_sol}"
            if is_contaminated(text):
                dropped_contamination += 1
                continue
            toks = tokenizer.encode(text, max_length=MAX_LEN, truncation=True)
            if len(toks) >= 96:
                samples.append(torch.tensor(toks, dtype=torch.long))
                stream_counts["contests"] += 1
            if stream_counts["contests"] >= contest_target:
                break
    except Exception as e:
        print(f"  Note: code_contests fallback to TACO hard split ({e})", flush=True)
        # Continue draining taco_iter to ensure fallback items are strictly disjoint from stream 1
        for item in taco_iter:
            if item.get("difficulty") in ["MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"]:
                sols_raw = item.get("solutions", "[]")
                solutions = json.loads(sols_raw) if isinstance(sols_raw, str) else sols_raw
                if solutions and isinstance(solutions, list):
                    code = solutions[0].strip()
                    if len(code) >= 40:
                        text = f"# Problem\n{item.get('question', '').strip()}\n\n# Solution\n{code}"
                        if is_contaminated(text):
                            dropped_contamination += 1
                            continue
                        toks = tokenizer.encode(text, max_length=MAX_LEN, truncation=True)
                        if len(toks) >= 96:
                            samples.append(torch.tensor(toks, dtype=torch.long))
                            stream_counts["contests"] += 1
            if stream_counts["contests"] >= contest_target:
                break

    # Build Diverse, Non-Repeating MBPP Pool across train, validation, and prompt splits
    print("  Indexing diverse MBPP pool (sanitized + full splits, zero test leakage)...", flush=True)
    mbpp_raw_items = []
    for cfg, splits in [("sanitized", ("train", "validation", "prompt")), ("full", ("train", "validation", "prompt"))]:
        for sp in splits:
            try:
                ds = load_dataset("google-research-datasets/mbpp", cfg, split=sp)
                mbpp_raw_items.extend(list(ds))
            except Exception:
                pass
    seen_ids = set()
    mbpp_unique = []
    for it in mbpp_raw_items:
        tid = it.get("task_id")
        if tid not in seen_ids:
            seen_ids.add(tid)
            mbpp_unique.append(it)
    random.Random(20261014).shuffle(mbpp_unique)
    half = len(mbpp_unique) // 2
    mbpp_a = mbpp_unique[:half]
    mbpp_b = mbpp_unique[half:]
    print(f"  Diverse MBPP items indexed: {len(mbpp_unique)} unique items (partitioned into {len(mbpp_a)} / {len(mbpp_b)}).", flush=True)

    # 3. MBPP Function Completion (Iterated once, zero loop repetition)
    print("  [3/5] Loading MBPP Stream (single pass)...", flush=True)
    for item in mbpp_a:
        prompt = item.get("prompt") or item.get("text", "")
        code = item.get("code", "")
        text = f'"""\n{prompt}\n"""\n{code}'
        if is_contaminated(text):
            dropped_contamination += 1
            continue
        toks = tokenizer.encode(text, max_length=MAX_LEN, truncation=True)
        if len(toks) >= 48:
            samples.append(torch.tensor(toks, dtype=torch.long))
            stream_counts["mbpp"] += 1

    # 4. General Python instruction replay (flytech/python-codes-25k)
    print("  [4/5] Loading General Python stream...", flush=True)
    py_target = int(target_count * 0.10)
    py_ds = load_dataset("flytech/python-codes-25k", split="train")
    for item in py_ds:
        code_text = item.get("text", "")
        if not code_text or is_contaminated(code_text):
            if code_text:
                dropped_contamination += 1
            continue
        toks = tokenizer.encode(code_text, max_length=MAX_LEN, truncation=True)
        if len(toks) >= 64:
            samples.append(torch.tensor(toks, dtype=torch.long))
            stream_counts["flytech"] += 1
        if stream_counts["flytech"] >= py_target:
            break

    # 5. Type-Annotated Function Implementation (Iterated once, zero loop repetition)
    print("  [5/5] Loading Function-Completion Stream (single pass)...", flush=True)
    for item in mbpp_b:
        text = f"# Function Implementation\n{item.get('code', '')}"
        if is_contaminated(text):
            dropped_contamination += 1
            continue
        toks = tokenizer.encode(text, max_length=MAX_LEN, truncation=True)
        if len(toks) >= 48:
            samples.append(torch.tensor(toks, dtype=torch.long))
            stream_counts["func_impl"] += 1

    # Backfill remaining target count from continuing taco_iter without repetition
    print(f"  Backfilling to {target_count} samples from remaining TACO stream...", flush=True)
    for item in taco_iter:
        if len(samples) >= target_count:
            break
        sols_raw = item.get("solutions", "[]")
        solutions = json.loads(sols_raw) if isinstance(sols_raw, str) else sols_raw
        if not solutions or not isinstance(solutions, list):
            continue
        code = solutions[0].strip()
        if len(code) < 40:
            continue
        text = f"# Problem\n{item.get('question', '').strip()}\n\n# Solution\n{code}"
        if is_contaminated(text):
            dropped_contamination += 1
            continue
        toks = tokenizer.encode(text, max_length=MAX_LEN, truncation=True)
        if len(toks) >= 96:
            samples.append(torch.tensor(toks, dtype=torch.long))
            stream_counts["taco"] += 1

    random.Random(20261014).shuffle(samples)

    # Hard guard against sample repetition
    assert len(samples) >= TOTAL_STEPS - START_STEP, f"Dataset short: {len(samples)} < {TOTAL_STEPS - START_STEP}; training would repeat samples"

    print(f"\n=== Final Dataset Yield Breakdown ===", flush=True)
    print(f"  TACO:              {stream_counts['taco']} samples ({stream_counts['taco']/len(samples)*100:.1f}%)")
    print(f"  Contests:          {stream_counts['contests']} samples ({stream_counts['contests']/len(samples)*100:.1f}%)")
    print(f"  MBPP:              {stream_counts['mbpp']} samples ({stream_counts['mbpp']/len(samples)*100:.1f}%)")
    print(f"  Flytech:           {stream_counts['flytech']} samples ({stream_counts['flytech']/len(samples)*100:.1f}%)")
    print(f"  Func Impl:         {stream_counts['func_impl']} samples ({stream_counts['func_impl']/len(samples)*100:.1f}%)")
    print(f"  Total Clean:       {len(samples)} samples")
    print(f"  Purged (13-gram):  {dropped_contamination} collisions", flush=True)
    if first_taco_preview:
        print(f"\n--- TACO Sample #1 Code Snippet Preview ---\n...{first_taco_preview}\n------------------------------------------", flush=True)
    return samples

def train_phase_c():
    torch.manual_seed(20261014)
    random.seed(20261014)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20261014)

    print(f"=== Starting Resumed Phase C (Steps {START_STEP}..{TOTAL_STEPS}) on {DEVICE} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    
    model = LoopSplitFrogNano(base_model, core_start=8, core_end=24, lora_rank=64).to(DEVICE)
    
    # Load Phase B Step 5000 checkpoint with weights_only=False
    print(f"Loading Phase B checkpoint: {RESUME_CKPT}...", flush=True)
    ckpt = torch.load(RESUME_CKPT, map_location=DEVICE, weights_only=False)
    model_state = model.state_dict()
    
    # Verify all LoRA and Gate parameters exist in checkpoint
    missing_lora = [k for k in model_state if "lora" in k and k not in ckpt["state_dict"]]
    assert not missing_lora, f"LoRA keys absent from checkpoint: {missing_lora[:5]}"
    
    missing_gates = [g for g in ("a2_mixer", "a2_mlp", "a3_mixer", "a3_mlp") if g not in ckpt["state_dict"]]
    assert not missing_gates, f"Gate keys absent from checkpoint: {missing_gates}"
    
    for k, v in ckpt["state_dict"].items():
        if k in model_state:
            model_state[k].copy_(v)
            
    # Verify gates loaded non-zero
    max_gate_val = max(model.a2_mixer.abs().max().item(), model.a3_mixer.abs().max().item())
    assert max_gate_val > 0.0, f"Gates loaded as all-zero ({max_gate_val}); resume state is invalid!"
    print(f"Checkpoint loaded successfully. Verified {len(model.lora_layers)} LoRA layers. Gate max: {max_gate_val:.4f}", flush=True)
    
    model.train()
    
    gate_params = [model.a2_mixer, model.a2_mlp, model.a3_mixer, model.a3_mlp]
    for gp in gate_params:
        gp.requires_grad = True
        
    lora_params = [p for p in model.parameters() if p.requires_grad and not any(p is gp for gp in gate_params)]
    
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": PEAK_LR_LORA, "weight_decay": 0.01},
        {"params": gate_params, "lr": PEAK_LR_GATES, "weight_decay": 0.0},
    ])
    
    loss_fn = HingeTrustRegionLoopSplitLoss(
        w2_start=0.3,
        w2_end=0.1,
        w3=1.0,
        lambda_kl=0.05,
        tau=0.05,
        anneal_steps=2000
    )
    
    train_tokens = build_upgraded_dataset(tokenizer, target_count=TOTAL_STEPS - START_STEP + 300)
    
    start_time = time.time()
    for step in range(START_STEP, TOTAL_STEPS + 1):
        step_offset = step - START_STEP
        cur_lr_lora = get_cosine_lr(step, START_STEP, TOTAL_STEPS, PEAK_LR_LORA, MIN_LR_LORA, WARMUP_STEPS)
        cur_lr_gates = get_cosine_lr(step, START_STEP, TOTAL_STEPS, PEAK_LR_GATES, MIN_LR_GATES, WARMUP_STEPS)
        optimizer.param_groups[0]["lr"] = cur_lr_lora
        optimizer.param_groups[1]["lr"] = cur_lr_gates
        
        input_ids = train_tokens[step_offset % len(train_tokens)].unsqueeze(0).to(DEVICE)
        
        optimizer.zero_grad()
        _, readouts = model(input_ids, num_loops=3, return_all_readouts=True, fixed_gate=None)
        
        loss_dict = loss_fn(readouts, input_ids, step_offset=step_offset)
        
        # Step 5001 Gate Check: two-sided check (must be active and <= 0.5)
        if step == START_STEP:
            print(f">>> STEP 5001 AUDIT: raw_kl = {loss_dict['raw_kl']:.6f} nats <<<", flush=True)
            assert 0.0 < loss_dict["raw_kl"] <= 0.5, f"Step 5001 raw_kl {loss_dict['raw_kl']} failed two-sided gate check (0.0 < raw_kl <= 0.5)!"
            
        loss = loss_dict["loss"]
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
        optimizer.step()
        
        with torch.no_grad():
            for gp in gate_params:
                gp.clamp_(-3.0, 3.0)
                
        if step % 50 == 0 or step == START_STEP:
            elapsed = time.time() - start_time
            steps_done = step_offset + 1
            sec_per_step = elapsed / max(1, steps_done)
            rem_min = ((TOTAL_STEPS - step) * sec_per_step) / 60.0
            
            g2_m = torch.tanh(model.a2_mixer).mean().item()
            g3_m = torch.tanh(model.a3_mixer).mean().item()
            b2_n = model.lora_layers[0].lora_B_2.norm().item()
            b3_n = model.lora_layers[0].lora_B_3.norm().item()
            
            print(f"Step {step:05d}/{TOTAL_STEPS} | "
                  f"Loss: {loss.item():.4f} | "
                  f"w2: {loss_dict['w2']:.3f} | "
                  f"CE Base: {loss_dict['ce_loop1_base']:.3f} | "
                  f"CE L2: {loss_dict['ce_loop2']:.3f} | "
                  f"CE L3: {loss_dict['ce_loop3']:.3f} | "
                  f"Raw KL: {loss_dict['raw_kl']:.4f} | "
                  f"Hinge KL: {loss_dict['hinge_kl']:.4f} | "
                  f"g2: {g2_m:.3f} | g3: {g3_m:.3f} | "
                  f"B2: {b2_n:.3f} | B3: {b3_n:.3f} | "
                  f"ETA: {rem_min:.1f}m ({sec_per_step:.2f}s/it)", flush=True)

        if step % 1000 == 0 or step == TOTAL_STEPS:
            ckpt_path = f"/home/ubuntu/loopsplit_phaseC_step{step}.pt"
            torch.save({
                "step": step,
                "state_dict": {
                    k: v for k, v in model.state_dict().items() 
                    if "lora" in k or k in {"a2_mixer", "a2_mlp", "a3_mixer", "a3_mlp"}
                }
            }, ckpt_path)
            print(f">>> Checkpoint saved: {ckpt_path} <<<", flush=True)

    print(f"\nPhase C completed in {(time.time() - start_time) / 60:.1f} minutes.", flush=True)

if __name__ == "__main__":
    train_phase_c()
