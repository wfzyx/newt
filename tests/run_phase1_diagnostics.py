import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from evalplus.data import get_human_eval_plus

from model.loopsplit_frognano import LoopSplitFrogNano

MODEL_ID = "microsoft/FrogNano-4B-2609"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
CHECKPOINT_PATH = "/home/ubuntu/loopsplit_step5000.pt"

STOP_SEQS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#"]

def truncate_at_stop(text):
    first_idx = len(text)
    for s in STOP_SEQS:
        pos = text.find(s)
        if pos != -1 and pos < first_idx:
            first_idx = pos
    return text[:first_idx]

def test_diagnostics():
    print("=== Running Phase 1 Advisor Diagnostics ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    base_model.eval()

    problems = get_human_eval_plus()
    task_keys = list(problems.keys())[:30]

    # -------------------------------------------------------------
    # DIAGNOSTIC 1: Decode-Path Consistency Check on HumanEval/0
    # -------------------------------------------------------------
    print("\n--- Diagnostic 1: Incremental Decode vs Full-Sequence Forward on Step 5000 ---", flush=True)
    model = LoopSplitFrogNano(base_model, core_start=8, core_end=24, lora_rank=64).to(DEVICE)
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

    p0 = problems["HumanEval/0"]["prompt"]
    inp = tokenizer.encode(p0, return_tensors="pt").to(DEVICE)
    curr = inp
    generated_tokens = []
    
    # Step-by-step greedy decode 32 tokens
    for _ in range(32):
        with torch.no_grad():
            logits = model(curr, num_loops=3)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr = torch.cat([curr, nxt], dim=-1)
            generated_tokens.append(nxt.item())

    # Now run ONE single full forward pass over the full sequence
    with torch.no_grad():
        full_logits = model(curr, num_loops=3)
        
    # Compare logits at each generated position
    prompt_len = inp.shape[1]
    divergences = 0
    max_logit_diff = 0.0
    for pos in range(len(generated_tokens)):
        sub_seq = curr[:, :prompt_len + pos]
        with torch.no_grad():
            step_logits = model(sub_seq, num_loops=3)[:, -1, :]
        full_step_logits = full_logits[:, prompt_len + pos - 1, :]
        
        diff = (step_logits - full_step_logits).abs().max().item()
        if diff > max_logit_diff:
            max_logit_diff = diff
        step_argmax = torch.argmax(step_logits, dim=-1).item()
        full_argmax = torch.argmax(full_step_logits, dim=-1).item()
        if step_argmax != full_argmax:
            divergences += 1

    print(f"Incremental vs Full Forward Max Logit Diff: {max_logit_diff:.8e}")
    print(f"Argmax Divergences across 32 tokens: {divergences}")
    if max_logit_diff < 1e-4:
        print(">> No cache/decode mismatch: forward pass is mathematically consistent.")
    else:
        print(">> WARNING: Incremental decode diverges from full forward pass!")

    # -------------------------------------------------------------
    # DIAGNOSTIC 2: Check Loop 1 Drift (Force g2=g3=0 on Step 5000)
    # -------------------------------------------------------------
    print("\n--- Diagnostic 2: Loop 1 Drift Check (g2=g3=0 on Step 5000) ---", flush=True)
    with torch.no_grad():
        model.g2_mixer.data.zero_()
        model.g2_mlp.data.zero_()
        model.g3_mixer.data.zero_()
        model.g3_mlp.data.zero_()

    passed_g0 = 0
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
        comp = tokenizer.decode(curr[0][inp.shape[1]:], skip_special_tokens=True)
        clean_comp = truncate_at_stop(comp)
        code = p + clean_comp + "\n" + test + f"\ncheck({entry})\n"
        try:
            exec(code, {})
            passed_g0 += 1
        except Exception:
            pass
    print(f"HumanEval Pass@1 with g2=g3=0 (Loop 1 alone): {passed_g0}/30 ({passed_g0/30*100:.1f}%)")

    # -------------------------------------------------------------
    # DIAGNOSTIC 3: Base FrogNano Baseline with Clean Stop Sequences
    # -------------------------------------------------------------
    print("\n--- Diagnostic 3: Pure Base FrogNano Baseline (with stop sequences) ---", flush=True)
    passed_base = 0
    for tid in task_keys:
        p = problems[tid]["prompt"]
        entry = problems[tid]["entry_point"]
        test = problems[tid]["test"]
        inp = tokenizer.encode(p, return_tensors="pt").to(DEVICE)
        out = base_model.generate(
            inp,
            max_new_tokens=96,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
        comp = tokenizer.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
        clean_comp = truncate_at_stop(comp)
        code = p + clean_comp + "\n" + test + f"\ncheck({entry})\n"
        try:
            exec(code, {})
            passed_base += 1
        except Exception:
            pass
    print(f"Pure Base FrogNano Pass@1 (30 tasks): {passed_base}/30 ({passed_base/30*100:.1f}%)")

if __name__ == "__main__":
    test_diagnostics()
