"""
Re-check of the best training-free cell against base on generation benchmarks, with per-problem outcomes
so the comparison is a paired (McNemar) test rather than two noisy pass counts.

    NEWT_HOME=/home/ubuntu python -u -m eval.tf_loop_recheck --cfg "8,11,2,anchor,block,0.5" --he-tokens 256

Runs HE-164 (greedy, --he-tokens new tokens) and MBPP-100 (greedy, 160 tokens) for base and for each --cfg.
Output JSON: per-task pass vectors, pass counts, discordant pairs (b = base-only pass, c = cfg-only pass),
exact binomial McNemar p-value.
"""
import os
import json
import time
import argparse
from math import comb

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.tf_loop import TFLoop
from model.presets import MODEL_ID
from eval import phase_d_harness as H

NEWT_HOME = os.environ.get("NEWT_HOME", "/home/ubuntu")
DEVICE = H.DEVICE


def parse_cfg(s):
    a, b, K, strat, mode, beta = s.split(",")
    return {"a": int(a), "b": int(b), "K": int(K), "strategy": strat, "mode": mode, "beta": float(beta)}


def he_vector(forward_fn, tok, tasks, max_new_tokens):
    out = []
    t0 = time.time()
    for i, t in enumerate(tasks, 1):
        comp = H.generate_greedy(forward_fn, tok, t["prompt"], max_new_tokens=max_new_tokens)
        code = t["prompt"] + comp + "\n\n" + t["test"] + f"\ncheck({t['entry_point']})\n"
        out.append(int(H.run_code(code)))
        if i % 40 == 0:
            print(f"    HE {i}/{len(tasks)} passed={sum(out)}", flush=True)
    return out, time.time() - t0


def mbpp_vector(forward_fn, tok, tasks):
    out = []
    t0 = time.time()
    for i, t in enumerate(tasks, 1):
        prompt = f'"""\n{t["prompt"]}\n{t["test_list"][0]}\n"""\n'
        comp = H.generate_greedy(forward_fn, tok, prompt, max_new_tokens=160)
        code = "\n".join(t.get("test_imports", [])) + "\n" + comp + "\n\n" + "\n".join(t["test_list"]) + "\n"
        out.append(int(H.run_code(code)))
        if i % 50 == 0:
            print(f"    MBPP {i}/{len(tasks)} passed={sum(out)}", flush=True)
    return out, time.time() - t0


def mcnemar(base_v, cfg_v):
    b = sum(1 for x, y in zip(base_v, cfg_v) if x and not y)   # base-only passes
    c = sum(1 for x, y in zip(base_v, cfg_v) if y and not x)   # cfg-only passes
    n = b + c
    k = min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"base_only": b, "cfg_only": c, "p_exact": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(NEWT_HOME, "phase_d_data.pt"))
    ap.add_argument("--cfg", action="append", default=[], help="a,b,K,strategy,mode,beta (repeatable)")
    ap.add_argument("--he-tokens", type=int, default=256)
    ap.add_argument("--out", default=os.path.join(NEWT_HOME, "tf_loop_recheck.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=H.DTYPE, device_map=DEVICE)
    base.eval()
    tf = TFLoop(base)
    data = torch.load(args.data, map_location="cpu", weights_only=False)["eval"]
    he_tasks, mbpp_tasks = data["humaneval"], data["mbpp100"]

    runs = [("base", lambda x: base(x).logits)]
    for s in args.cfg:
        c = parse_cfg(s)
        name = f"[{c['a']},{c['b']}]_{c['strategy']}_K{c['K']}_{c['mode']}_b{c['beta']}"
        runs.append((name, tf.make_forward(**c)))

    res = {"he_tokens": args.he_tokens, "runs": {}}
    for name, fn in runs:
        print(f"== {name}", flush=True)
        he_v, he_s = he_vector(fn, tok, he_tasks, args.he_tokens)
        mb_v, mb_s = mbpp_vector(fn, tok, mbpp_tasks)
        res["runs"][name] = {"he": he_v, "he_pass": sum(he_v), "he_sec": he_s,
                             "mbpp": mb_v, "mbpp_pass": sum(mb_v), "mbpp_sec": mb_s}
        print(f"   HE-{len(he_v)} {sum(he_v)}  ({he_s/60:.1f} min)   MBPP-{len(mb_v)} {sum(mb_v)}  ({mb_s/60:.1f} min)", flush=True)
        if name != "base":
            b = res["runs"]["base"]
            res["runs"][name]["mcnemar_he"] = mcnemar(b["he"], he_v)
            res["runs"][name]["mcnemar_mbpp"] = mcnemar(b["mbpp"], mb_v)
            print(f"   McNemar HE {res['runs'][name]['mcnemar_he']}  MBPP {res['runs'][name]['mcnemar_mbpp']}", flush=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
