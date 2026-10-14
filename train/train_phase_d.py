"""
Phase D: steps 5001..7000, resumed from Phase B step 5000.

    NEWT_HOME=/home/ubuntu python -u -m train.train_phase_d [--smoke]

Per optimizer step: 16 micro-batches (batch 1, <=512 tokens) in a fixed 8:5:3 stream order
(general replay : docstring functions : competitive-as-functions), gradient accumulation, then
gate L2 restoring force, clip 1.0, AdamW step, hard clamp |a| <= 2.0.

Loss per micro-batch (train/phase_d_loss.py):
    0.25*CE_L2 + 0.75*CE_L3 + lambda_kl[s] * relu(KL(p_L1||p_L3) - tau[s])
tau[s] = KL_0[s] measured at step 5000 on a calibration slice (64/40/24 samples) before any update.

Cadence: quick suite every 250 steps (ID TACO NLL, OOD pretrain NLL, OOD func NLL, HE-40),
full suite + checkpoint every 500 (HE-164, MBPP-100). Kill criteria K1-K4 evaluated after each suite.
"""
import os
import sys
import json
import math
import time
import random
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.loopsplit_frognano import LoopSplitFrogNano
from train.phase_d_loss import PhaseDLoss
from eval import phase_d_harness as H

MODEL_ID = "microsoft/FrogNano-4B-2609"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
NEWT_HOME = os.environ.get("NEWT_HOME", "/home/ubuntu")
SEED = 20261015

START_STEP = 5001
TOTAL_STEPS = 7000
ACCUM = {1: 8, 2: 5, 3: 3}          # micro-batches per step, by stream
MICRO_PER_STEP = sum(ACCUM.values())  # 16

PEAK_LR_LORA = 5e-5
PEAK_LR_GATES = 5e-4
WARMUP = 300
MIN_LR_FRAC = 0.10
WD_LORA = 0.01
CLIP_NORM = 1.0

LAMBDA_KL = {1: 0.5, 2: 0.5, 3: 0.1}
LAMBDA_G = 2e-3
GATE_CLAMP = 2.0
W2, W3 = 0.25, 0.75

CALIB = {1: 64, 2: 40, 3: 24}       # samples per stream for KL_0 measurement
QUICK_EVERY = 250
FULL_EVERY = 500

# Kill thresholds
K1_HE164_DROP = 3.0                 # points below step-5000 baseline
K2_OOD_PRETRAIN_WORSEN = 0.010      # nats
K2_MODE = "per_window"              # "per_window": each of 2 consecutive windows worsens > thr (strict)
                                    # "cumulative": monotone worsening and (c - a) > thr over the 2-window span
K3_TACO_IMPROVE = 0.030             # nats, with OOD-func worsening
K4_GATE_MIN_ABS = 1.0
K4_LORA_REF = (3.031, 2.688)        # layer-0 lora_B_2 / lora_B_3 Frobenius at step 5000
K4_LORA_SHIFT = 0.25

RESUME_CKPT = os.path.join(NEWT_HOME, "loopsplit_phaseB_step5000.pt")
DATA_PATH = os.path.join(NEWT_HOME, "phase_d_data.pt")
LOG_PATH = os.path.join(NEWT_HOME, "phase_d_log.jsonl")


# --------------------------------------------------------------------------- schedule
def lr_at(step: int, peak: float) -> float:
    off = step - START_STEP
    if off < WARMUP:
        return peak * off / WARMUP
    prog = (off - WARMUP) / max(1, (TOTAL_STEPS - START_STEP) - WARMUP)
    return peak * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * prog)))


# --------------------------------------------------------------------------- data
class StreamSampler:
    """Deterministic 8:5:3 micro-batch order per step; each stream is a shuffled no-repeat queue."""

    def __init__(self, train: dict, seed: int):
        self.rng = random.Random(seed)
        self.queues = {k: list(v) for k, v in train.items()}
        for q in self.queues.values():
            self.rng.shuffle(q)
        self.ptr = {k: 0 for k in self.queues}
        self.wrapped = {k: 0 for k in self.queues}
        # interleave so streams are spread across the accumulation window
        self.order = [1, 2, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 2, 1]
        assert sorted(self.order) == sorted(sum(([k] * n for k, n in ACCUM.items()), []))

    def next_micro(self, stream: int) -> dict:
        q = self.queues[stream]
        if self.ptr[stream] >= len(q):
            self.ptr[stream] = 0
            self.wrapped[stream] += 1
            self.rng.shuffle(q)
        s = q[self.ptr[stream]]
        self.ptr[stream] += 1
        return s

    def step_batch(self):
        return [self.next_micro(k) for k in self.order]


# --------------------------------------------------------------------------- kill criteria
class KillCriteria:
    def __init__(self, baseline: dict):
        self.base = baseline
        self.hist = []  # quick-suite results in order

    def check(self, res: dict, is_full: bool) -> list:
        fired = []
        self.hist.append(res)
        if is_full and "he164" in res and res["he164"] < self.base["he164"] - K1_HE164_DROP:
            fired.append(f"K1 he164 {res['he164']:.1f} < base {self.base['he164']:.1f} - {K1_HE164_DROP}")
        if len(self.hist) >= 3:
            a, b, c = (h["nll_ood_pretrain"] for h in self.hist[-3:])
            if K2_MODE == "per_window":
                k2 = (b - a) > K2_OOD_PRETRAIN_WORSEN and (c - b) > K2_OOD_PRETRAIN_WORSEN
            else:
                k2 = b > a and c > b and (c - a) > K2_OOD_PRETRAIN_WORSEN
            if k2:
                fired.append(f"K2[{K2_MODE}] ood_pretrain worsened over two windows: {a:.4f} -> {b:.4f} -> {c:.4f}")
        if (self.base["nll_taco"] - res["nll_taco"]) > K3_TACO_IMPROVE and res["nll_ood_func"] > self.base["nll_ood_func"]:
            fired.append(f"K3 taco -{self.base['nll_taco'] - res['nll_taco']:.4f} while ood_func +{res['nll_ood_func'] - self.base['nll_ood_func']:.4f}")
        if res["a_min_abs"] < K4_GATE_MIN_ABS:
            fired.append(f"K4 gate min|a|={res['a_min_abs']:.3f} < {K4_GATE_MIN_ABS}")
        for key, ref in (("B2_l0", K4_LORA_REF[0]), ("B3_l0", K4_LORA_REF[1])):
            if abs(res[key] - ref) / ref > K4_LORA_SHIFT:
                fired.append(f"K4 {key}={res[key]:.3f} shifted >{K4_LORA_SHIFT*100:.0f}% from {ref}")
        return fired


# --------------------------------------------------------------------------- main
def save_ckpt(model, step, tau, path):
    torch.save({
        "step": step,
        "phase": "D",
        "tau": tau,
        "a2_mixer": model.a2_mixer.detach().cpu(), "a2_mlp": model.a2_mlp.detach().cpu(),
        "a3_mixer": model.a3_mixer.detach().cpu(), "a3_mlp": model.a3_mlp.detach().cpu(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()
                       if "lora" in k or k in {"a2_mixer", "a2_mlp", "a3_mixer", "a3_mlp"}},
    }, path)
    print(f">>> checkpoint saved: {path}", flush=True)


def log(rec: dict):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="4 steps, tiny calib, quick eval on 8 HE tasks")
    ap.add_argument("--no-kill", action="store_true", help="log kill criteria but do not stop")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    total_steps = START_STEP + 3 if args.smoke else TOTAL_STEPS
    calib = {k: 2 for k in CALIB} if args.smoke else CALIB
    he_quick = 8 if args.smoke else 40

    print(f"=== Phase D {START_STEP}..{total_steps} on {DEVICE} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    model = LoopSplitFrogNano(base, core_start=8, core_end=24, lora_rank=64).to(DEVICE)

    ckpt = H.load_checkpoint(model, RESUME_CKPT)
    assert ckpt.get("step") == 5000, f"expected step-5000 checkpoint, got {ckpt.get('step')}"
    st = H.gate_lora_stats(model)
    print(f"resumed: g2={st['g2_mean']:.3f} g3={st['g3_mean']:.3f} |a|min={st['a_min_abs']:.3f} "
          f"B2_l0={st['B2_l0']:.3f} B3_l0={st['B3_l0']:.3f}", flush=True)
    assert st["a_min_abs"] > 0.0, "gates loaded as zero"

    data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    train, data_eval = data["train"], data["eval"]
    counts = {k: len(v) for k, v in train.items()}
    print(f"data: {counts} | decontam dropped {data['meta'].get('decontam_dropped')}", flush=True)
    for k, n in ACCUM.items():
        need = n * (total_steps - START_STEP + 1)
        if len(train[k]) < need:
            print(f"WARN stream{k}: {len(train[k])} samples < {need} needed; will repeat", flush=True)

    gate_params = [model.a2_mixer, model.a2_mlp, model.a3_mixer, model.a3_mlp]
    for g in gate_params:
        g.requires_grad = True
    lora_params = [p for p in model.parameters() if p.requires_grad and not any(p is g for g in gate_params)]
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": PEAK_LR_LORA, "weight_decay": WD_LORA},
        {"params": gate_params, "lr": PEAK_LR_GATES, "weight_decay": 0.0},
    ])
    loss_fn = PhaseDLoss(w2=W2, w3=W3, lambda_kl=LAMBDA_KL, lambda_g=LAMBDA_G)
    forward_fn = lambda x: model(x, num_loops=3)  # noqa: E731

    def run_micro(sample):
        ids = sample["input_ids"].unsqueeze(0).to(DEVICE)
        mask = sample["loss_mask"].unsqueeze(0).to(DEVICE)
        _, readouts = model(ids, num_loops=3, return_all_readouts=True)
        return loss_fn(readouts, ids, mask, sample["stream"])

    # ---- tau = KL_0 per stream, measured before any update
    model.eval()
    tau = {}
    with torch.no_grad():
        for k, n in calib.items():
            kls = [run_micro(s)["raw_kl"] for s in train[k][:n]]
            tau[k] = float(sum(kls) / len(kls))
    loss_fn.set_tau(tau)
    print(f"KL_0 (tau) per stream: { {k: round(v, 5) for k, v in tau.items()} }", flush=True)
    assert all(0.0 < v <= 0.5 for v in tau.values()), f"KL_0 out of expected range: {tau}"

    # ---- step-5000 baseline (full suite) -> kill reference
    print("baseline (step 5000) full suite...", flush=True)
    t0 = time.time()
    if args.smoke:
        baseline = H.quick_suite(model, forward_fn, tokenizer, data_eval, he_n=he_quick)
        baseline["he164"] = baseline[f"he{he_quick}"]
    else:
        baseline = H.full_suite(model, forward_fn, tokenizer, data_eval)
    baseline["step"] = 5000
    baseline["kind"] = "baseline"
    print(f"baseline: {H.fmt(baseline)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    log(baseline)
    kill = KillCriteria(baseline)

    sampler = StreamSampler(train, SEED)
    model.train()
    t_start = time.time()
    agg_keys = ("ce_loop1_base", "ce_loop2", "ce_loop3", "raw_kl", "hinge_kl")

    for step in range(START_STEP, total_steps + 1):
        lr_l, lr_g = lr_at(step, PEAK_LR_LORA), lr_at(step, PEAK_LR_GATES)
        optimizer.param_groups[0]["lr"] = lr_l
        optimizer.param_groups[1]["lr"] = lr_g
        optimizer.zero_grad(set_to_none=True)

        agg = {k: {kk: 0.0 for kk in agg_keys} for k in ACCUM}
        total = 0.0
        for sample in sampler.step_batch():
            d = run_micro(sample)
            (d["loss"] / MICRO_PER_STEP).backward()
            total += d["loss"].item() / MICRO_PER_STEP
            for kk in agg_keys:
                agg[sample["stream"]][kk] += d[kk] / ACCUM[sample["stream"]]

        gate_pen = loss_fn.gate_penalty(gate_params)
        gate_pen.backward()
        total += gate_pen.item()

        gn_l = torch.nn.utils.clip_grad_norm_(lora_params, CLIP_NORM).item()
        gn_g = torch.nn.utils.clip_grad_norm_(gate_params, CLIP_NORM).item()
        optimizer.step()
        with torch.no_grad():
            for g in gate_params:
                g.clamp_(-GATE_CLAMP, GATE_CLAMP)

        if step == START_STEP:
            print(f">>> step {step} audit: loss={total:.4f} gate_pen={gate_pen.item():.4f} gn_lora={gn_l:.3f} gn_gate={gn_g:.3f}", flush=True)

        if step % 25 == 0 or step == START_STEP:
            el = time.time() - t_start
            spi = el / (step - START_STEP + 1)
            st = H.gate_lora_stats(model)
            s1, s2, s3 = agg[1], agg[2], agg[3]
            print(f"S{step} loss={total:.4f} lr={lr_l:.2e}/{lr_g:.2e} | "
                  f"s1 ce1/3={s1['ce_loop1_base']:.3f}/{s1['ce_loop3']:.3f} kl={s1['raw_kl']:.4f} | "
                  f"s2 ce1/3={s2['ce_loop1_base']:.3f}/{s2['ce_loop3']:.3f} kl={s2['raw_kl']:.4f} | "
                  f"s3 ce1/3={s3['ce_loop1_base']:.3f}/{s3['ce_loop3']:.3f} kl={s3['raw_kl']:.4f} | "
                  f"|a|min={st['a_min_abs']:.3f} g2={st['g2_mean']:.3f} g3={st['g3_mean']:.3f} "
                  f"B2={st['B2_l0']:.3f} B3={st['B3_l0']:.3f} gn={gn_l:.2f}/{gn_g:.2f} | "
                  f"ETA {(total_steps - step) * spi / 60:.0f}m ({spi:.1f}s/step)", flush=True)
            log({"kind": "train", "step": step, "loss": total, "gate_pen": gate_pen.item(), "lr_lora": lr_l, "lr_gate": lr_g,
                 "gn_lora": gn_l, "gn_gate": gn_g, "streams": agg, **st})

        is_full = step % FULL_EVERY == 0 or step == total_steps
        is_quick = step % QUICK_EVERY == 0 or (args.smoke and step == total_steps)
        if is_quick or is_full:
            t0 = time.time()
            if is_full and not args.smoke:
                res = H.full_suite(model, forward_fn, tokenizer, data_eval)
                save_ckpt(model, step, tau, os.path.join(NEWT_HOME, f"loopsplit_phaseD_step{step}.pt"))
            else:
                res = H.quick_suite(model, forward_fn, tokenizer, data_eval, he_n=he_quick)
                if args.smoke:
                    res["he164"] = res[f"he{he_quick}"]
            res.update({"kind": "eval", "step": step, "full": is_full, "eval_min": (time.time() - t0) / 60})
            print(f"EVAL S{step}: {H.fmt(res)} ({res['eval_min']:.1f} min)", flush=True)
            fired = kill.check(res, is_full)
            res["kill"] = fired
            log(res)
            model.train()
            if fired:
                print(f"!!! KILL CRITERIA FIRED at step {step}: {fired}", flush=True)
                if not args.no_kill:
                    save_ckpt(model, step, tau, os.path.join(NEWT_HOME, f"loopsplit_phaseD_killed_step{step}.pt"))
                    sys.exit(2)

    print(f"Phase D done in {(time.time() - t_start) / 60:.1f} min; stream wraps: {sampler.wrapped}", flush=True)


if __name__ == "__main__":
    main()
