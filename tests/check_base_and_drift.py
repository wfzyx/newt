import json
import torch
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

def run_checks():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    base_model.eval()

    problems = get_human_eval_plus()
    task_keys = list(problems.keys())[:30]

    # 1. Base FrogNano Baseline
    base_passed = 0
    for tid in task_keys:
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
    print(f"Pure Base FrogNano Pass@1 (30 tasks): {base_passed}/30 ({base_passed/30*100:.1f}%)")

    # 2. Step 5000 with g2=g3=0
    model = LoopSplitFrogNano(base_model, core_start=8, core_end=24, lora_rank=64).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model_state = model.state_dict()
    for k, v in ckpt["state_dict"].items():
        if k in model_state:
            model_state[k].copy_(v)
    model.g2_mixer.data.zero_()
    model.g2_mlp.data.zero_()
    model.g3_mixer.data.zero_()
    model.g3_mlp.data.zero_()
    model.eval()

    g0_passed = 0
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
        comp = truncate_at_stop(tokenizer.decode(curr[0][inp.shape[1]:], skip_special_tokens=True))
        code = p + comp + "\n" + test + "\ncheck(" + entry + ")\n"
        try:
            exec(code, {})
            g0_passed += 1
        except:
            pass
    print(f"Step 5000 with g2=g3=0 Pass@1 (30 tasks): {g0_passed}/30 ({g0_passed/30*100:.1f}%)")

if __name__ == "__main__":
    run_checks()
