import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evalplus.data import get_human_eval_plus
from model.loopsplit_frognano import LoopSplitFrogNano

MODEL_ID = "microsoft/FrogNano-4B-2609"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
CHECKPOINT_PATH = "/home/ubuntu/loopsplit_corrected_step5000.pt"
STOP_SEQS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#"]

def truncate_at_stop(text):
    first_idx = len(text)
    for s in STOP_SEQS:
        pos = text.find(s)
        if pos != -1 and pos < first_idx:
            first_idx = pos
    return text[:first_idx]

def run_30_tasks():
    print("=== Running 30-Task HumanEval Benchmark Comparison ===", flush=True)
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

    problems = get_human_eval_plus()
    task_keys = list(problems.keys())[:30]

    # 1. Base FrogNano Evaluation
    print("\n[1/2] Evaluating Pure Base FrogNano...", flush=True)
    base_passed = 0
    t0 = time.time()
    for idx, tid in enumerate(task_keys, 1):
        p = problems[tid]["prompt"]
        entry = problems[tid]["entry_point"]
        test = problems[tid]["test"]
        inp = tokenizer.encode(p, return_tensors="pt").to(DEVICE)
        out = base_model.generate(inp, max_new_tokens=96, do_sample=False, eos_token_id=tokenizer.eos_token_id)
        comp = truncate_at_stop(tokenizer.decode(out[0][inp.shape[1]:], skip_special_tokens=True))
        code = p + comp + "\n" + test + "\ncheck(" + entry + ")\n"
        try:
            exec(code, {})
            base_passed += 1
        except:
            pass
        if idx % 10 == 0:
            print(f"  Base progress: {idx}/30 (running passed: {base_passed})", flush=True)
    base_acc = (base_passed / 30) * 100
    print(f"Pure Base FrogNano Pass@1 (30 tasks): {base_passed}/30 ({base_acc:.1f}%) in {time.time() - t0:.1f}s", flush=True)

    # 2. Corrected LoopSplit Step 5000 Evaluation
    print("\n[2/2] Evaluating Corrected LoopSplit Step 5000 (Loop 3 active)...", flush=True)
    loop_passed = 0
    t1 = time.time()
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
            loop_passed += 1
        except:
            pass
        if idx % 10 == 0:
            print(f"  LoopSplit progress: {idx}/30 (running passed: {loop_passed})", flush=True)
    loop_acc = (loop_passed / 30) * 100
    print(f"Corrected LoopSplit Step 5000 Pass@1 (30 tasks): {loop_passed}/30 ({loop_acc:.1f}%) in {time.time() - t1:.1f}s", flush=True)

    print("\n" + "=" * 60)
    print(f"Pure Base FrogNano:                {base_passed}/30 ({base_acc:.1f}%)")
    print(f"Corrected LoopSplit Step 5000:     {loop_passed}/30 ({loop_acc:.1f}%)")
    print("=" * 60, flush=True)

if __name__ == "__main__":
    run_30_tasks()
