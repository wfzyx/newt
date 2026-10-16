"""
Phase D evaluation harness.

Used in-process by train/train_phase_d.py at the 250/500-step cadence, and standalone:

    python -m eval.phase_d_harness --ckpt /home/ubuntu/loopsplit_phaseB_step5000.pt --full
    python -m eval.phase_d_harness --base                # pure 1-pass base, no checkpoint
    python -m eval.phase_d_harness --ckpt X --quick      # NLL slices + HE-40 only

Metrics
  nll_taco        ID TACO-test NLL (all tokens; same construction as run_checkpoint_trend_eval.py)
  nll_ood_pretrain  held-out general Python windows (all tokens)
  nll_ood_func    held-out docstring functions (body tokens only)
  he{N}           greedy pass@1 on first N HumanEval tasks (original HumanEval tests, exec in subprocess)
  mbpp100         greedy pass@1 on first 100 MBPP-sanitized test tasks
  gates / lora    min|a|, mean tanh(a), layer-0 lora_B_2 / lora_B_3 Frobenius norms

Generation matches the Phase B/C scripts (greedy, 96 new tokens, stop sequences), but test execution
runs in a subprocess with a 10s timeout instead of exec() in-process.
"""
import os
import sys
import json
import math
import time
import argparse
import tempfile
import subprocess

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.loopsplit_frognano import LoopSplitFrogNano
from model.presets import MODEL_ID, CORE_START, CORE_END, LORA_RANK

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
NEWT_HOME = os.environ.get("NEWT_HOME", "/home/ubuntu")
STOP_SEQS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#"]
EOS_IDS = {151643, 151645}


def truncate_at_stop(text: str) -> str:
    cut = len(text)
    for s in STOP_SEQS:
        i = text.find(s)
        if i != -1 and i < cut:
            cut = i
    return text[:cut]


# --------------------------------------------------------------------------- model io
def load_checkpoint(model: LoopSplitFrogNano, path: str) -> dict:
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = model.state_dict()
    n = 0
    for k, v in ckpt["state_dict"].items():
        if k in state:
            state[k].copy_(v)
            n += 1
    for g in ("a2_mixer", "a2_mlp", "a3_mixer", "a3_mlp"):
        if g in ckpt:
            getattr(model, g).data.copy_(ckpt[g])
    assert n > 0, f"no tensors loaded from {path}"
    return ckpt


def gate_lora_stats(model: LoopSplitFrogNano) -> dict:
    gates = [model.a2_mixer, model.a2_mlp, model.a3_mixer, model.a3_mlp]
    act = torch.tanh if getattr(model, "gate_mode", "tanh") == "tanh" else (lambda x: x)
    with torch.no_grad():
        out = {
            "a_min_abs": min(g.abs().min().item() for g in gates),
            "a_max_abs": max(g.abs().max().item() for g in gates),
            "g2_mean": act(model.a2_mixer).mean().item(),
            "g3_mean": act(model.a3_mixer).mean().item(),
            "g2_mlp_mean": act(model.a2_mlp).mean().item(),
            "g3_mlp_mean": act(model.a3_mlp).mean().item(),
            "B2_l0": model.lora_layers[0].lora_B_2.float().norm().item(),
            "B3_l0": model.lora_layers[0].lora_B_3.float().norm().item(),
            "B2_all": math.sqrt(sum(l.lora_B_2.float().norm().item() ** 2 for l in model.lora_layers)),
            "B3_all": math.sqrt(sum(l.lora_B_3.float().norm().item() ** 2 for l in model.lora_layers)),
        }
        if getattr(model, "boundary_adapter", False):
            out["loop_gate"] = [v.item() for v in model.loop_gate]
            out["proj_dev_from_I"] = (model.loop_proj.weight.float() - torch.eye(model.loop_proj.weight.shape[0], device=model.loop_proj.weight.device)).norm().item()
            out["step_emb_norm"] = [v.float().norm().item() for v in model.loop_step_emb]
        if getattr(model, "entry_mode", "none") == "concat":
            out["concat_alpha"] = model.concat_alpha.item()
            out["concat_beta"] = model.concat_beta.item()
            out["concat_BA"] = (model.concat_B.float() @ model.concat_A.float()).norm().item()
            out["concat_scale"] = model.concat_scale.item()
        if getattr(model, "entry_mode", "none") == "reinject":
            out["entry_w"] = [torch.sigmoid(v).item() for v in model.entry_mix]
        return out


# --------------------------------------------------------------------------- NLL
@torch.no_grad()
def eval_nll(forward_fn, samples, masked: bool = False) -> float:
    """Token-weighted NLL across docs. samples: list of LongTensor or list of {input_ids, loss_mask}."""
    tot, n = 0.0, 0.0
    for s in samples:
        if isinstance(s, dict):
            ids, mask = s["input_ids"], s["loss_mask"]
        else:
            ids, mask = s, torch.ones_like(s, dtype=torch.bool)
        inp = ids.unsqueeze(0).to(DEVICE)
        logits = forward_fn(inp)[:, :-1, :].float()
        tgt = inp[:, 1:]
        m = mask[1:].to(DEVICE).float() if masked else torch.ones_like(tgt, dtype=torch.float).view(-1)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none")
        tot += (ce * m.view(-1)).sum().item()
        n += m.sum().item()
    return tot / max(n, 1.0)


# --------------------------------------------------------------------------- generation
@torch.no_grad()
def generate_greedy(forward_fn, tokenizer, prompt: str, max_new_tokens: int = 96) -> str:
    inp = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    curr = inp
    for _ in range(max_new_tokens):
        logits = forward_fn(curr)
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        curr = torch.cat([curr, nxt], dim=-1)
        if nxt.item() in EOS_IDS or nxt.item() == tokenizer.eos_token_id:
            break
        txt = tokenizer.decode(curr[0][inp.shape[1]:])
        if any(s in txt for s in STOP_SEQS):
            break
    return truncate_at_stop(tokenizer.decode(curr[0][inp.shape[1]:], skip_special_tokens=True))


def run_code(code: str, timeout: float = 10.0) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


def eval_humaneval(forward_fn, tokenizer, tasks, n: int, log_every: int = 40) -> dict:
    passed = 0
    t0 = time.time()
    tasks = tasks[:n]
    for i, t in enumerate(tasks, 1):
        comp = generate_greedy(forward_fn, tokenizer, t["prompt"])
        code = t["prompt"] + comp + "\n\n" + t["test"] + f"\ncheck({t['entry_point']})\n"
        passed += run_code(code)
        if i % log_every == 0:
            print(f"    HE {i}/{len(tasks)} passed={passed}", flush=True)
    return {"passed": passed, "total": len(tasks), "pct": 100.0 * passed / max(1, len(tasks)), "sec": time.time() - t0}


def eval_mbpp(forward_fn, tokenizer, tasks, log_every: int = 50) -> dict:
    """Prompt format mirrors Phase B/C training: docstring-style task text, then the model writes the code."""
    passed = 0
    t0 = time.time()
    for i, t in enumerate(tasks, 1):
        prompt = f'"""\n{t["prompt"]}\n{t["test_list"][0]}\n"""\n'
        comp = generate_greedy(forward_fn, tokenizer, prompt, max_new_tokens=160)
        code = "\n".join(t.get("test_imports", [])) + "\n" + comp + "\n\n" + "\n".join(t["test_list"]) + "\n"
        passed += run_code(code)
        if i % log_every == 0:
            print(f"    MBPP {i}/{len(tasks)} passed={passed}", flush=True)
    return {"passed": passed, "total": len(tasks), "pct": 100.0 * passed / max(1, len(tasks)), "sec": time.time() - t0}


# --------------------------------------------------------------------------- suites
def quick_suite(model, forward_fn, tokenizer, data_eval: dict, he_n: int = 40) -> dict:
    was_training = model.training
    model.eval()
    out = {
        "nll_taco": eval_nll(forward_fn, data_eval["taco_test"]),
        "nll_ood_pretrain": eval_nll(forward_fn, data_eval["ood_pretrain"]),
        "nll_ood_func": eval_nll(forward_fn, data_eval["ood_func"], masked=True),
    }
    he = eval_humaneval(forward_fn, tokenizer, data_eval["humaneval"], he_n)
    out[f"he{he_n}"] = he["pct"]
    out[f"he{he_n}_passed"] = he["passed"]
    out.update(gate_lora_stats(model))
    if was_training:
        model.train()
    return out


def full_suite(model, forward_fn, tokenizer, data_eval: dict) -> dict:
    out = quick_suite(model, forward_fn, tokenizer, data_eval, he_n=40)
    model.eval()
    he = eval_humaneval(forward_fn, tokenizer, data_eval["humaneval"], 164)
    mb = eval_mbpp(forward_fn, tokenizer, data_eval["mbpp100"])
    out.update({"he164": he["pct"], "he164_passed": he["passed"], "mbpp100": mb["pct"], "mbpp100_passed": mb["passed"]})
    return out


def fmt(res: dict) -> str:
    keys = ["nll_taco", "nll_ood_pretrain", "nll_ood_func", "he40", "he164", "mbpp100", "a_min_abs", "g2_mean", "g3_mean", "B2_l0", "B3_l0"]
    return " | ".join(f"{k}={res[k]:.4f}" if isinstance(res.get(k), float) else f"{k}={res.get(k)}" for k in keys if k in res)


# --------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--base", action="store_true", help="evaluate pure base (1-pass HF forward)")
    ap.add_argument("--data", default=os.path.join(NEWT_HOME, "phase_d_data.pt"))
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    base.eval()
    data = torch.load(args.data, map_location="cpu", weights_only=False)["eval"]

    model = LoopSplitFrogNano(base, core_start=CORE_START, core_end=CORE_END, lora_rank=LORA_RANK).to(DEVICE)
    if args.base:
        forward_fn = lambda x: base(x).logits  # noqa: E731
        name = "base"
    else:
        assert args.ckpt, "--ckpt or --base required"
        load_checkpoint(model, args.ckpt)
        forward_fn = lambda x: model(x, num_loops=3)  # noqa: E731
        name = os.path.basename(args.ckpt)
    model.eval()

    t0 = time.time()
    res = full_suite(model, forward_fn, tokenizer, data) if args.full and not args.quick else quick_suite(model, forward_fn, tokenizer, data)
    res["name"] = name
    res["sec"] = time.time() - t0
    print(f"\n[{name}] {fmt(res)}  ({res['sec']/60:.1f} min)")
    out = args.out or os.path.join(NEWT_HOME, f"phase_d_eval_{name}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
