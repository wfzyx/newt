"""
Training-free loop sweep on the frozen base (no checkpoint, no training).

Stage 1  grid over (window, K, strategy, mode) on a doc subset of the three NLL slices; paired per-doc
         CE delta vs the 1-pass base. Partial results are flushed to --out after every config.
Stage 2  top --refine configs re-run on the full 150-doc slices with bootstrap 95% CIs.
Stage 3  HE-164 greedy pass@1 for the base and the top --he-top configs.

    NEWT_HOME=/home/ubuntu python -u -m eval.tf_loop_sweep --sweep-docs 60 --refine 8 --he-top 3

Windows follow arXiv:2605.23872: short windows (n=2..6) at 40-80% depth. [8,23] (our LoopSplit core) is
included as the control for "16-layer window = past the cliff".
"""
import os
import json
import time
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.tf_loop import TFLoop
from model.presets import MODEL_ID
from eval import phase_d_harness as H
from eval.stage0_loop_diag import boot_ci

NEWT_HOME = os.environ.get("NEWT_HOME", "/home/ubuntu")
DEVICE = H.DEVICE
SLICES = [("taco_test", False), ("ood_pretrain", False), ("ood_func", True)]


@torch.no_grad()
def per_doc_ce(forward_fn, samples, masked):
    out = []
    for s in samples:
        if isinstance(s, dict):
            ids, mask = s["input_ids"], s["loss_mask"]
        else:
            ids, mask = s, torch.ones_like(s, dtype=torch.bool)
        inp = ids.unsqueeze(0).to(DEVICE)
        tgt = inp[:, 1:].reshape(-1)
        m = mask[1:].to(DEVICE).float() if masked else torch.ones_like(tgt, dtype=torch.float)
        lg = forward_fn(inp)[:, :-1, :].float()
        lg = lg.reshape(-1, lg.size(-1))
        ce = F.cross_entropy(lg, tgt, reduction="none")
        out.append(((ce * m).sum() / m.sum().clamp_min(1)).item())
    return out


def cfg_name(c):
    s = f"[{c['a']},{c['b']}]_{c['strategy']}_K{c['K']}_{c['mode']}"
    if c["strategy"] == "anchor":
        s += f"_b{c['beta']}"
    return s


def build_grid(num_layers, control=(8, 23)):
    grid = []
    lo, hi = int(0.15 * num_layers), int(0.85 * num_layers)  # windows inside ~15-85% depth
    windows = []
    for n in (2, 4, 6):
        for a in range(lo, hi - n + 2, 2):
            windows.append((a, a + n - 1))
    windows.append(control)
    block_strats = [("naive", 2, 0.5), ("euler", 2, 0.5), ("euler", 3, 0.5), ("anchor", 2, 0.5), ("uniform", 2, 0.5)]
    for (a, b) in windows:
        for strat, K, beta in block_strats:
            grid.append({"a": a, "b": b, "K": K, "strategy": strat, "mode": "block", "beta": beta})
        grid.append({"a": a, "b": b, "K": 2, "strategy": "euler", "mode": "layer", "beta": 0.5})
    return grid


def score(res):
    """Ranking metric: mean paired delta over the three slices (lower = better)."""
    return sum(res[s]["delta"] for s, _ in SLICES) / len(SLICES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(NEWT_HOME, "phase_d_data.pt"))
    ap.add_argument("--out", default=os.path.join(NEWT_HOME, "tf_loop_sweep.json"))
    ap.add_argument("--sweep-docs", type=int, default=60)
    ap.add_argument("--refine", type=int, default=8)
    ap.add_argument("--he-top", type=int, default=3)
    ap.add_argument("--he-n", type=int, default=164)
    ap.add_argument("--only", default=None, help="semicolon-separated config names to run instead of the grid")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=H.DTYPE, device_map=DEVICE)
    base.eval()
    tf = TFLoop(base)
    data = torch.load(args.data, map_location="cpu", weights_only=False)["eval"]
    base_fn = lambda x: base(x).logits  # noqa: E731

    # exactness check: K=1 must equal the base forward
    probe = data["taco_test"][0]
    probe = (probe["input_ids"] if isinstance(probe, dict) else probe)[:128].unsqueeze(0).to(DEVICE)
    d = (tf.forward(probe, 8, 23, K=1) - base_fn(probe)).abs().max().item()
    print(f"K=1 parity max|dlogits| = {d:.3e}", flush=True)
    assert d < 1e-2, "wrapper does not reproduce base at K=1"

    results = {"model": MODEL_ID, "sweep_docs": args.sweep_docs, "base": {}, "sweep": {}, "refine": {}, "he": {}}

    def flush():
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # ---------------------------------------------------------------- base per-doc CE (full slices)
    base_ce = {}
    t0 = time.time()
    for s, masked in SLICES:
        base_ce[s] = per_doc_ce(base_fn, data[s], masked)
        results["base"][s] = sum(base_ce[s]) / len(base_ce[s])
    print(f"base: " + " ".join(f"{s}={results['base'][s]:.4f}" for s, _ in SLICES) + f"  ({time.time()-t0:.0f}s)", flush=True)
    flush()

    def evaluate(cfg, ndocs):
        fn = tf.make_forward(a=cfg["a"], b=cfg["b"], K=cfg["K"], strategy=cfg["strategy"], mode=cfg["mode"], beta=cfg["beta"])
        res = {}
        for s, masked in SLICES:
            ce = per_doc_ce(fn, data[s][:ndocs], masked)
            deltas = [c - b for c, b in zip(ce, base_ce[s][:ndocs])]
            lo, hi = boot_ci(deltas, n=2000)
            res[s] = {"nll": sum(ce) / len(ce), "delta": sum(deltas) / len(deltas), "ci": [lo, hi],
                      "win": sum(1 for d in deltas if d < 0) / len(deltas)}
        return res

    def line(name, res):
        return f"{name:36s} " + " ".join(f"{s[:4]} {res[s]['delta']:+.4f}[{res[s]['ci'][0]:+.3f},{res[s]['ci'][1]:+.3f}]" for s, _ in SLICES) + f"  score {score(res):+.4f}"

    # ---------------------------------------------------------------- stage 1: sweep
    grid = build_grid(tf.num_layers)
    if args.only:
        want = set(args.only.split(";"))
        grid = [c for c in grid if cfg_name(c) in want]
    print(f"stage 1: {len(grid)} configs x {args.sweep_docs} docs/slice", flush=True)
    for i, cfg in enumerate(grid, 1):
        name = cfg_name(cfg)
        t1 = time.time()
        res = evaluate(cfg, args.sweep_docs)
        res["cfg"] = cfg
        res["sec"] = time.time() - t1
        results["sweep"][name] = res
        print(f"[{i}/{len(grid)}] {line(name, res)} ({res['sec']:.0f}s)", flush=True)
        flush()

    ranked = sorted(results["sweep"].items(), key=lambda kv: score(kv[1]))
    print("\nstage 1 top 10:")
    for name, res in ranked[:10]:
        print("  " + line(name, res))

    # ---------------------------------------------------------------- stage 2: refine on full slices
    top = [name for name, _ in ranked[: args.refine]]
    print(f"\nstage 2: {len(top)} configs on full slices", flush=True)
    for name in top:
        cfg = results["sweep"][name]["cfg"]
        res = evaluate(cfg, 10**9)
        res["cfg"] = cfg
        results["refine"][name] = res
        print("  " + line(name, res), flush=True)
        flush()
    ranked2 = sorted(results["refine"].items(), key=lambda kv: score(kv[1]))

    # ---------------------------------------------------------------- stage 3: HE-164
    he_tasks = data["humaneval"]
    print(f"\nstage 3: HE-{args.he_n} on base + top {args.he_top}", flush=True)
    r = H.eval_humaneval(base_fn, tok, he_tasks, args.he_n)
    results["he"]["base"] = r
    print(f"  base  HE {r['passed']}/{r['total']} ({r['sec']/60:.1f} min)", flush=True)
    flush()
    for name, _ in ranked2[: args.he_top]:
        cfg = results["refine"][name]["cfg"]
        fn = tf.make_forward(a=cfg["a"], b=cfg["b"], K=cfg["K"], strategy=cfg["strategy"], mode=cfg["mode"], beta=cfg["beta"])
        r = H.eval_humaneval(fn, tok, he_tasks, args.he_n)
        results["he"][name] = r
        print(f"  {name:36s} HE {r['passed']}/{r['total']} ({r['sec']/60:.1f} min)", flush=True)
        flush()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
