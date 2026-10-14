"""
Stage 0: does loop 3 beat loop 1 (exact base) anywhere outside TACO? No training.

For each checkpoint, one forward per doc with return_all_readouts=True gives loop-1/2/3 logits,
so CE_L1 (base), CE_L2, CE_L3 come from the same pass. Reported per slice:
  taco_test (ID), ood_pretrain (general Python windows), ood_func (docstring-function bodies)
plus per-doc paired deltas (L3 - L1) with a bootstrap 95% CI so a 0.01-nat gap is not over-read.

Then a gate ablation on the chosen checkpoint: HE-40 at num_loops=1 (=base), 2, 3.

    NEWT_HOME=/home/ubuntu python -u -m eval.stage0_loop_diag
"""
import os
import json
import time
import random
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.loopsplit_frognano import LoopSplitFrogNano
from eval import phase_d_harness as H

NEWT_HOME = os.environ.get("NEWT_HOME", "/home/ubuntu")
DEVICE = H.DEVICE

CKPTS = [
    ("A-1500", "loopsplit_phaseA_step1500.pt"),
    ("B-5000", "loopsplit_phaseB_step5000.pt"),
    ("C-6000", "loopsplit_phaseC_step6000.pt"),
    ("C-10000", "loopsplit_phaseC_step10000.pt"),
]
SLICES = [("taco_test", False), ("ood_pretrain", False), ("ood_func", True)]


@torch.no_grad()
def per_loop_ce(model, samples, masked: bool):
    """Returns dict loop-> list of per-doc token-mean CE (same docs, same tokens, so deltas are paired)."""
    out = {1: [], 2: [], 3: []}
    for s in samples:
        if isinstance(s, dict):
            ids, mask = s["input_ids"], s["loss_mask"]
        else:
            ids, mask = s, torch.ones_like(s, dtype=torch.bool)
        inp = ids.unsqueeze(0).to(DEVICE)
        tgt = inp[:, 1:].reshape(-1)
        m = (mask[1:].to(DEVICE).float() if masked else torch.ones_like(tgt, dtype=torch.float))
        _, ro = model(inp, num_loops=3, return_all_readouts=True)
        for k in (1, 2, 3):
            lg = ro[f"logits_loop_{k}"][:, :-1, :].float().reshape(-1, ro[f"logits_loop_{k}"].size(-1))
            ce = F.cross_entropy(lg, tgt, reduction="none")
            out[k].append(((ce * m).sum() / m.sum().clamp_min(1)).item())
    return out


def boot_ci(deltas, n=5000, seed=0):
    rng = random.Random(seed)
    N = len(deltas)
    means = []
    for _ in range(n):
        means.append(sum(deltas[rng.randrange(N)] for _ in range(N)) / N)
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(NEWT_HOME, "phase_d_data.pt"))
    ap.add_argument("--ablate", default="B-5000", help="checkpoint name for the HE-40 loops=1/2/3 ablation ('' to skip)")
    ap.add_argument("--he-n", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(NEWT_HOME, "stage0_results.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(H.MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(H.MODEL_ID, torch_dtype=H.DTYPE, device_map=DEVICE)
    base.eval()
    model = LoopSplitFrogNano(base, core_start=8, core_end=24, lora_rank=64).to(DEVICE)
    model.eval()
    ev = torch.load(args.data, map_location="cpu", weights_only=False)["eval"]
    for name, _ in SLICES:
        print(f"slice {name}: {len(ev[name])} docs", flush=True)

    results = {}
    for name, fn in CKPTS:
        path = os.path.join(NEWT_HOME, fn)
        if not os.path.exists(path):
            print(f"skip {name}: {path} missing", flush=True)
            continue
        H.load_checkpoint(model, path)
        st = H.gate_lora_stats(model)
        t0 = time.time()
        rec = {"gates": st, "slices": {}}
        print(f"\n== {name}  g2={st['g2_mean']:.3f} g3={st['g3_mean']:.3f} B2_l0={st['B2_l0']:.3f} B3_l0={st['B3_l0']:.3f}", flush=True)
        for sl, masked in SLICES:
            ce = per_loop_ce(model, ev[sl], masked)
            mean = {k: sum(v) / len(v) for k, v in ce.items()}
            d31 = [a - b for a, b in zip(ce[3], ce[1])]
            d21 = [a - b for a, b in zip(ce[2], ce[1])]
            lo, hi = boot_ci(d31)
            wins = sum(1 for d in d31 if d < 0) / len(d31)
            rec["slices"][sl] = {"ce_l1": mean[1], "ce_l2": mean[2], "ce_l3": mean[3],
                                 "d31_mean": sum(d31) / len(d31), "d31_ci95": [lo, hi], "d31_win_frac": wins,
                                 "d21_mean": sum(d21) / len(d21)}
            verdict = "L3<L1 (CI excludes 0)" if hi < 0 else ("L3>L1 (CI excludes 0)" if lo > 0 else "no sig diff")
            print(f"  {sl:13s} L1={mean[1]:.4f} L2={mean[2]:.4f} L3={mean[3]:.4f} | d31={sum(d31)/len(d31):+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}] win={wins:.2f} | {verdict}", flush=True)
        rec["sec"] = time.time() - t0
        results[name] = rec

    if args.ablate and args.ablate in dict(CKPTS):
        H.load_checkpoint(model, os.path.join(NEWT_HOME, dict(CKPTS)[args.ablate]))
        abl = {}
        print(f"\n== HE-{args.he_n} loop ablation on {args.ablate}", flush=True)
        for L in (1, 2, 3):
            r = H.eval_humaneval(lambda x, L=L: model(x, num_loops=L), tok, ev["humaneval"], args.he_n, log_every=1000)
            abl[f"loops{L}"] = r
            print(f"  num_loops={L}: {r['passed']}/{r['total']} ({r['pct']:.1f}%) in {r['sec']:.0f}s", flush=True)
        results[f"ablation_{args.ablate}"] = abl

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
