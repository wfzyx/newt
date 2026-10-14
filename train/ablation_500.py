"""
Diagnostic B: does multi-pass recurrence transfer OOD better than a plain 1-pass LoRA at matched compute?

Arms (all from base init, identical data order / accumulation, CE-only loss):
  control    : num_loops=1, single_pass_lora=True  (LoRA_2 on the single core pass, no gates)
  treatment  : num_loops=3, tanh gates learnable from a=2.0 (Phase B recipe), 0.25*CE_L2 + 0.75*CE_L3
  salvage    : num_loops=3, raw zero-init gates + identity-init boundary projection + zero-init step
               embeddings (e_0 = 0 fixed), gate group LR 5e-3 wd 0. Bit-parity with base at step 0.

Compute matching: 3-loop step ~2.4x a 1-pass step (8.6 vs 3.6 s measured), so compare
  salvage --steps 500   vs   control --steps 1200.

After training: paired per-doc CE delta vs base on taco_test / ood_pretrain / ood_func (bootstrap 95% CI),
then HE-40.

    NEWT_HOME=/home/ubuntu python -u -m train.ablation_500 --arm control --steps 1200
    NEWT_HOME=/home/ubuntu python -u -m train.ablation_500 --arm salvage --steps 500
"""
import os
import json
import math
import time
import random
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.loopsplit_frognano import LoopSplitFrogNano
from train.train_phase_d import StreamSampler, ACCUM, MICRO_PER_STEP
from eval import phase_d_harness as H
from eval.stage0_loop_diag import boot_ci

NEWT_HOME = os.environ.get("NEWT_HOME", "/home/ubuntu")
DEVICE = H.DEVICE
SEED = 20261016

STEPS = 500          # overridden by --steps
WARMUP = 50
PEAK_LR_LORA = 5e-5
PEAK_LR_GATES = 5e-4         # treatment (tanh gates)
PEAK_LR_GATES_SALVAGE = 5e-3  # salvage (raw gates + loop_gate)
WD_LORA = 0.01
CLIP = 1.0
GATE_INIT = 2.0
GATE_CLAMP = 3.0
W2, W3 = 0.25, 0.75


def lr_at(step: int, peak: float) -> float:
    if step < WARMUP:
        return peak * step / WARMUP
    p = (step - WARMUP) / max(1, STEPS - WARMUP)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def masked_ce(logits, ids, mask):
    lg = logits[:, :-1, :].float()
    tgt = ids[:, 1:].reshape(-1)
    m = mask[:, 1:].reshape(-1).float()
    ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt, reduction="none")
    return (ce * m).sum() / m.sum().clamp_min(1)


@torch.no_grad()
def paired_eval(model, arm: str, ev: dict) -> dict:
    model.eval()
    out = {}
    for sl, masked in (("taco_test", False), ("ood_pretrain", False), ("ood_func", True)):
        deltas, ce_b, ce_m = [], [], []
        for s in ev[sl]:
            ids = (s["input_ids"] if isinstance(s, dict) else s).unsqueeze(0).to(DEVICE)
            mask = (s["loss_mask"] if isinstance(s, dict) else torch.ones_like(ids[0], dtype=torch.bool)).unsqueeze(0).to(DEVICE)
            if not masked:
                mask = torch.ones_like(ids, dtype=torch.bool)
            if arm != "control":
                _, ro = model(ids, num_loops=3, return_all_readouts=True)
                b = masked_ce(ro["logits_loop_1"], ids, mask).item()
                mdl = masked_ce(ro["logits_loop_3"], ids, mask).item()
            else:
                b = masked_ce(model(ids, num_loops=1), ids, mask).item()
                mdl = masked_ce(model(ids, num_loops=1, single_pass_lora=True), ids, mask).item()
            ce_b.append(b)
            ce_m.append(mdl)
            deltas.append(mdl - b)
        lo, hi = boot_ci(deltas)
        out[sl] = {"ce_base": sum(ce_b) / len(ce_b), "ce_model": sum(ce_m) / len(ce_m),
                   "delta": sum(deltas) / len(deltas), "ci95": [lo, hi],
                   "win_frac": sum(d < 0 for d in deltas) / len(deltas)}
        print(f"  {sl:13s} base={out[sl]['ce_base']:.4f} model={out[sl]['ce_model']:.4f} "
              f"delta={out[sl]['delta']:+.4f} [{lo:+.4f},{hi:+.4f}] win={out[sl]['win_frac']:.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["control", "treatment", "salvage", "reinject"], required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--data", default=os.path.join(NEWT_HOME, "phase_d_data_ablation2.pt"))
    ap.add_argument("--eval-data", default=os.path.join(NEWT_HOME, "phase_d_data.pt"), help="150-doc eval slices (Stage 0 file)")
    ap.add_argument("--he-n", type=int, default=40)
    args = ap.parse_args()
    arm = args.arm
    global STEPS
    STEPS = args.steps
    tag = f"{arm}_{STEPS}"

    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    tok = AutoTokenizer.from_pretrained(H.MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(H.MODEL_ID, torch_dtype=H.DTYPE, device_map=DEVICE)
    if arm == "salvage":
        model = LoopSplitFrogNano(base, core_start=8, core_end=24, lora_rank=64, gate_mode="raw", boundary_adapter=True).to(DEVICE)
    elif arm == "reinject":
        model = LoopSplitFrogNano(base, core_start=8, core_end=24, lora_rank=64, entry_mode="reinject", entry_init=0.99).to(DEVICE)
    else:
        model = LoopSplitFrogNano(base, core_start=8, core_end=24, lora_rank=64).to(DEVICE)
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    train = data["train"]
    ev = torch.load(args.eval_data, map_location="cpu", weights_only=False)["eval"]
    print(f"[{arm}] eval slices: { {k: len(ev[k]) for k in ('taco_test', 'ood_pretrain', 'ood_func')} }", flush=True)
    print(f"[{arm}] data: { {k: len(v) for k, v in train.items()} }", flush=True)
    for k, n in ACCUM.items():
        assert len(train[k]) >= n * STEPS, f"stream{k}: {len(train[k])} < {n * STEPS}"

    gate_params = [model.a2_mixer, model.a2_mlp, model.a3_mixer, model.a3_mlp]
    if arm == "salvage":
        gate_params.append(model.loop_gate)
    if arm == "reinject":
        gate_params = [model.entry_mix]   # per-layer gates unused in this mode
        for g in (model.a2_mixer, model.a2_mlp, model.a3_mixer, model.a3_mlp):
            g.requires_grad = False
    if arm == "treatment":
        with torch.no_grad():
            for g in gate_params:
                g.fill_(GATE_INIT)
    for g in gate_params:
        g.requires_grad = arm != "control"
    lora_params = [p for p in model.parameters() if p.requires_grad and not any(p is g for g in gate_params)]
    if arm == "reinject":
        # shared LoRA_2 across loops 2/3; LoRA_3 unused
        for l in model.lora_layers:
            l.lora_A_3.requires_grad = False
            l.lora_B_3.requires_grad = False
        lora_params = [p for p in lora_params if p.requires_grad]
    if arm == "control":
        # only LoRA_2 is used; freeze LoRA_3 so parameter counts are honest (identical active params to one loop)
        for l in model.lora_layers:
            l.lora_A_3.requires_grad = False
            l.lora_B_3.requires_grad = False
        lora_params = [p for p in lora_params if p.requires_grad]
    groups = [{"params": lora_params, "lr": PEAK_LR_LORA, "weight_decay": WD_LORA}]
    if arm == "treatment":
        groups.append({"params": gate_params, "lr": PEAK_LR_GATES, "weight_decay": 0.0})
    elif arm in ("salvage", "reinject"):
        groups.append({"params": gate_params, "lr": PEAK_LR_GATES_SALVAGE, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    n_train = sum(p.numel() for p in lora_params)
    print(f"[{tag}] trainable LoRA/adapter params: {n_train/1e6:.1f}M (+{sum(g.numel() for g in gate_params) if arm != 'control' else 0} gate scalars)", flush=True)

    if arm in ("salvage", "reinject"):
        # Step-0 parity: salvage must be exact; reinject starts at 99% clean input so a small gap is expected (logged)
        model.eval()
        with torch.no_grad():
            ids0 = train[1][0]["input_ids"].unsqueeze(0).to(DEVICE)
            _, ro0 = model(ids0, num_loops=3, return_all_readouts=True)
            diff = (ro0["logits_loop_3"] - ro0["logits_loop_1"]).abs().max().item()
            ce1 = masked_ce(ro0["logits_loop_1"], ids0, torch.ones_like(ids0, dtype=torch.bool)).item()
            ce3 = masked_ce(ro0["logits_loop_3"], ids0, torch.ones_like(ids0, dtype=torch.bool)).item()
        print(f"[{tag}] step-0 parity max|logits_L3 - logits_L1| = {diff:.3e}  CE_L1={ce1:.4f} CE_L3={ce3:.4f}", flush=True)
        if arm == "salvage":
            assert diff < 1e-3, "salvage arm is not base-identical at step 0"

    sampler = StreamSampler(train, SEED)
    model.train()
    t0 = time.time()
    log_path = os.path.join(NEWT_HOME, f"ablation_{arm}_log.jsonl")
    for step in range(1, STEPS + 1):
        lr = lr_at(step, PEAK_LR_LORA)
        opt.param_groups[0]["lr"] = lr
        if len(opt.param_groups) > 1:
            opt.param_groups[1]["lr"] = lr_at(step, PEAK_LR_GATES if arm == "treatment" else PEAK_LR_GATES_SALVAGE)
        opt.zero_grad(set_to_none=True)
        tot, per = 0.0, {1: 0.0, 2: 0.0, 3: 0.0}
        for s in sampler.step_batch():
            ids = s["input_ids"].unsqueeze(0).to(DEVICE)
            mask = s["loss_mask"].unsqueeze(0).to(DEVICE)
            if arm != "control":
                _, ro = model(ids, num_loops=3, return_all_readouts=True)
                loss = W2 * masked_ce(ro["logits_loop_2"], ids, mask) + W3 * masked_ce(ro["logits_loop_3"], ids, mask)
            else:
                loss = masked_ce(model(ids, num_loops=1, single_pass_lora=True), ids, mask)
            (loss / MICRO_PER_STEP).backward()
            tot += loss.item() / MICRO_PER_STEP
            per[s["stream"]] += loss.item() / ACCUM[s["stream"]]
        gn = torch.nn.utils.clip_grad_norm_(lora_params, CLIP).item()
        if arm != "control":
            torch.nn.utils.clip_grad_norm_(gate_params, CLIP)
        opt.step()
        if arm == "treatment":
            with torch.no_grad():
                for g in gate_params:
                    g.clamp_(-GATE_CLAMP, GATE_CLAMP)
        if step % 25 == 0 or step == 1:
            st = H.gate_lora_stats(model)
            spi = (time.time() - t0) / step
            extra = f" lg={st['loop_gate'][0]:+.3f}/{st['loop_gate'][1]:+.3f} projdev={st['proj_dev_from_I']:.3f}" if arm == "salvage" else ""
            if arm == "reinject":
                extra = f" w={st['entry_w'][0]:.4f}/{st['entry_w'][1]:.4f}"
            print(f"[{tag}] S{step} loss={tot:.4f} s1={per[1]:.3f} s2={per[2]:.3f} s3={per[3]:.3f} lr={lr:.2e} gn={gn:.2f} "
                  f"g2={st['g2_mean']:+.3f}/{st['g2_mlp_mean']:+.3f} g3={st['g3_mean']:+.3f}/{st['g3_mlp_mean']:+.3f} "
                  f"B2={st['B2_l0']:.3f} B3={st['B3_l0']:.3f}{extra} "
                  f"ETA {(STEPS-step)*spi/60:.0f}m ({spi:.1f}s/step)", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": tot, "per": per, "lr": lr, "gn": gn, **st}) + "\n")

    ckpt_path = os.path.join(NEWT_HOME, f"ablation_{tag}.pt")
    keep = set(model.adapter_param_names())
    torch.save({"step": STEPS, "arm": arm, "gate_mode": model.gate_mode, "boundary_adapter": model.boundary_adapter,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}}, ckpt_path)
    print(f"[{tag}] trained in {(time.time()-t0)/60:.1f} min -> {ckpt_path}", flush=True)

    print(f"\n[{tag}] paired eval vs base", flush=True)
    res = {"arm": arm, "steps": STEPS, "nll": paired_eval(model, arm, ev), "gates": H.gate_lora_stats(model)}
    fwd = (lambda x: model(x, num_loops=1, single_pass_lora=True)) if arm == "control" else (lambda x: model(x, num_loops=3))
    he = H.eval_humaneval(fwd, tok, ev["humaneval"], args.he_n, log_every=1000)
    res[f"he{args.he_n}"] = he
    print(f"[{tag}] HE-{args.he_n}: {he['passed']}/{he['total']} ({he['pct']:.1f}%)", flush=True)
    out = os.path.join(NEWT_HOME, f"ablation_{tag}_results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
