"""
Qwen3.5-0.8B retrofit study: does the literature recipe work at a budget where it was shown to work?

    NEWT_MODEL=qwen08 NEWT_HOME=/home/ubuntu python -u -m train.study_qwen08 --arm A|B|C [--tokens 100e6]

Arms (same data order, same LoRA LR/schedule, same token budget):
  A  control   : 1 pass, LoRA r=64 on core projections, frozen base.                  (the bar)
  B  reinject  : 3 loops, z_t = w*p + (1-w)*h_{t-1} (w0 = 0.99, own group lr 5e-3), shared LoRA, frozen base.
  C  unfrozen  : as B, plus the core layers (6..17) fully trainable at lr 1e-5 (no LoRA on them).
  D  concat    : as C but the loop entry is u_t = RMSNorm(W [h_{t-1}; p]) (arXiv:2511.07384), W init [0.05 I | I].

Looped arms also get:
  - smoothed exit loss: L = sum_t lam_t CE_t over t in {2,3}; lam shifts linearly from (0.7, 0.3) to (0.25, 0.75)
    over the first half of training so every loop trains and the last loop ends dominant. (2610.00673)
  - recurrence ramp: T=2 only until CE_2 < CE_1 on a 64-sample probe (checked every 250 steps), then T=3.
    Hard-opens T=3 at 30% of training regardless. (2511.07384)

Budget: --tokens / (16 micro-batches * 512) steps. 100M tokens = 12,207 steps.
Eval: quick suite (3 NLL slices + HE-40) every 1000 steps, full suite at the end; paired per-doc delta vs loop-1
(= base) with bootstrap CI at the end. Checkpoint every 2000 steps + end.
"""
import os
import sys
import json
import math
import time
import random
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.loopsplit_frognano import LoopSplitFrogNano
from model.presets import MODEL_ID, CORE_START, CORE_END, LORA_RANK, NAME as PRESET
from train.train_phase_d import StreamSampler, ACCUM, MICRO_PER_STEP
from train.ablation_500 import masked_ce, paired_eval
from eval import phase_d_harness as H

NEWT_HOME = os.environ.get("NEWT_HOME", "/home/ubuntu")
DEVICE = H.DEVICE
SEED = 20261017
TOK_PER_STEP = MICRO_PER_STEP * 512

LR_LORA = 5e-5
LR_MIX = 5e-3
LR_BRIDGE = 5e-5   # concat projection is a dense 2d x d matrix: keep it at the LoRA LR (5e-4 diverged in smoke)
LR_CORE = 1e-5
WD = 0.01
CLIP = 1.0
WARMUP_FRAC = 0.03
MIN_LR_FRAC = 0.10

QUICK_EVERY = 1000
CKPT_EVERY = 2000
RAMP_CHECK_EVERY = 250
RAMP_HARD_OPEN_FRAC = 0.30


def lr_at(step, total, peak):
    w = max(1, int(total * WARMUP_FRAC))
    if step < w:
        return peak * step / w
    p = (step - w) / max(1, total - w)
    return peak * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * p)))


def exit_weights(step, total):
    """(lam2, lam3): (0.7,0.3) -> (0.25,0.75) linearly over the first half, then fixed."""
    f = min(1.0, step / max(1, total * 0.5))
    return 0.7 + (0.25 - 0.7) * f, 0.3 + (0.75 - 0.3) * f


@torch.no_grad()
def probe_ce2_lt_ce1(model, samples):
    model.eval()
    d = []
    for s in samples:
        ids = s["input_ids"].unsqueeze(0).to(DEVICE)
        mask = s["loss_mask"].unsqueeze(0).to(DEVICE)
        _, ro = model(ids, num_loops=2, return_all_readouts=True)
        d.append(masked_ce(ro["logits_loop_2"], ids, mask).item() - masked_ce(ro["logits_loop_1"], ids, mask).item())
    model.train()
    return sum(d) / len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C", "D"], required=True)
    ap.add_argument("--tokens", type=float, default=100e6)
    ap.add_argument("--data", default=os.path.join(NEWT_HOME, "q08_train.pt"))
    ap.add_argument("--eval-data", default=os.path.join(NEWT_HOME, "q08_eval.pt"))
    ap.add_argument("--he-n", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bs", type=int, default=4, help="micro-batch size; 16 samples/step regardless")
    args = ap.parse_args()
    arm = args.arm
    total = 30 if args.smoke else int(args.tokens / TOK_PER_STEP)
    tag = f"q08_{arm}_{total}"
    assert PRESET == "qwen08", "run with NEWT_MODEL=qwen08"

    torch.manual_seed(SEED); random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=H.DTYPE, device_map=DEVICE)
    looped = arm in ("B", "C", "D")
    entry = {"A": "none", "B": "reinject", "C": "reinject", "D": "concat"}[arm]
    model = LoopSplitFrogNano(base, core_start=CORE_START, core_end=CORE_END, lora_rank=LORA_RANK,
                              entry_mode=entry, entry_init=0.99 if entry == "reinject" else 0.05).to(DEVICE)
    print(f"[{tag}] {MODEL_ID} layers={model.num_layers} core={CORE_START}..{CORE_END-1} steps={total} "
          f"tokens/step={TOK_PER_STEP}", flush=True)

    train = torch.load(args.data, map_location="cpu", weights_only=False)["train"]
    ev = torch.load(args.eval_data, map_location="cpu", weights_only=False)["eval"]
    print(f"[{tag}] train={ {k: len(v) for k, v in train.items()} } eval={ {k: len(ev[k]) for k in ('taco_test','ood_pretrain','ood_func')} }", flush=True)

    # ---- parameter groups
    for g in (model.a2_mixer, model.a2_mlp, model.a3_mixer, model.a3_mlp):
        g.requires_grad = False
    for l in model.lora_layers:
        l.lora_A_3.requires_grad = False
        l.lora_B_3.requires_grad = False
    groups = []
    if arm in ("C", "D"):
        # full-weight core; drop LoRA on core (redundant). Base layers 0..5, 18..23 stay frozen.
        for l in model.lora_layers:
            l.lora_A_2.requires_grad = False
            l.lora_B_2.requires_grad = False
        core_params = []
        for i in range(CORE_START, CORE_END):
            for p in model.backbone.layers[i].parameters():
                p.requires_grad = True
                core_params.append(p)
        groups.append({"params": core_params, "lr": LR_CORE, "weight_decay": WD, "name": "core"})
    else:
        lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
        groups.append({"params": lora_params, "lr": LR_LORA, "weight_decay": WD, "name": "lora"})
    if arm in ("B", "C"):
        groups.append({"params": [model.entry_mix], "lr": LR_MIX, "weight_decay": 0.0, "name": "mix"})
    elif arm == "D":
        groups.append({"params": [model.concat_A, model.concat_B, model.concat_bias] + list(model.concat_norm.parameters()),
                       "lr": LR_LORA, "weight_decay": WD, "name": "bridge"})
        groups.append({"params": [model.concat_alpha, model.concat_beta, model.concat_scale],
                       "lr": LR_MIX, "weight_decay": 0.0, "name": "mix"})
    opt = torch.optim.AdamW(groups)
    n_tr = sum(p.numel() for g in groups for p in g["params"])
    print(f"[{tag}] trainable params: {n_tr/1e6:.1f}M in groups {[g['name'] for g in groups]}", flush=True)

    # ---- step-0 check
    model.eval()
    with torch.no_grad():
        ids0 = train[1][0]["input_ids"].unsqueeze(0).to(DEVICE)
        m0 = torch.ones_like(ids0, dtype=torch.bool)
        if looped:
            _, ro0 = model(ids0, num_loops=3, return_all_readouts=True)
            print(f"[{tag}] step-0 CE_L1={masked_ce(ro0['logits_loop_1'], ids0, m0).item():.4f} "
                  f"CE_L2={masked_ce(ro0['logits_loop_2'], ids0, m0).item():.4f} CE_L3={masked_ce(ro0['logits_loop_3'], ids0, m0).item():.4f}", flush=True)
        else:
            a = masked_ce(model(ids0, num_loops=1), ids0, m0).item()
            b = masked_ce(model(ids0, num_loops=1, single_pass_lora=True), ids0, m0).item()
            print(f"[{tag}] step-0 CE base={a:.4f} lora={b:.4f} (must match)", flush=True)
            assert abs(a - b) < 1e-3

    def collate(samples):
        """Right-pad to the longest; pads get loss_mask=0. Causal attention + causal DeltaNet mean real-token
        logits are identical to the unbatched case, so this changes only throughput."""
        L = max(len(s["input_ids"]) for s in samples)
        ids = torch.full((len(samples), L), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros((len(samples), L), dtype=torch.bool)
        for i, s in enumerate(samples):
            n = len(s["input_ids"])
            ids[i, :n] = s["input_ids"]
            mask[i, :n] = s["loss_mask"]
        return ids.to(DEVICE), mask.to(DEVICE)

    def fwd_train(ids, mask, T, lam):
        if not looped:
            return masked_ce(model(ids, num_loops=1, single_pass_lora=True), ids, mask), {}
        _, ro = model(ids, num_loops=T, return_all_readouts=True)
        ce = {t: masked_ce(ro[f"logits_loop_{t}"], ids, mask) for t in range(1, T + 1)}
        if T == 2:
            loss = ce[2]
        else:
            loss = lam[0] * ce[2] + lam[1] * ce[3]
        return loss, {t: v.item() for t, v in ce.items()}

    sampler = StreamSampler(train, SEED)
    probe = train[2][:64]
    model.train()
    T = 2 if looped else 1
    ramp_open_step = None
    t0 = time.time()
    log_path = os.path.join(NEWT_HOME, f"{tag}_log.jsonl")
    for step in range(1, total + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, {"lora": LR_LORA, "core": LR_CORE, "mix": LR_MIX, "bridge": LR_LORA}[g["name"]])
        lam = exit_weights(step, total)
        opt.zero_grad(set_to_none=True)
        tot, agg = 0.0, {}
        batch = sampler.step_batch()
        n_micro = math.ceil(len(batch) / args.bs)
        for i in range(0, len(batch), args.bs):
            ids, mask = collate(batch[i:i + args.bs])
            loss, ce = fwd_train(ids, mask, T, lam)
            (loss / n_micro).backward()
            tot += loss.item() / n_micro
            for t, v in ce.items():
                agg[t] = agg.get(t, 0.0) + v / n_micro
        gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], CLIP).item()
        opt.step()

        # recurrence ramp
        if looped and T == 2 and (step % RAMP_CHECK_EVERY == 0 or step >= total * RAMP_HARD_OPEN_FRAC):
            d21 = probe_ce2_lt_ce1(model, probe)
            if d21 < 0 or step >= total * RAMP_HARD_OPEN_FRAC:
                T = 3
                ramp_open_step = step
                print(f"[{tag}] S{step} ramp: opening T=3 (probe CE2-CE1={d21:+.4f})", flush=True)
            else:
                print(f"[{tag}] S{step} ramp: hold T=2 (probe CE2-CE1={d21:+.4f})", flush=True)

        if step % 50 == 0 or step == 1:
            spi = (time.time() - t0) / step
            st = H.gate_lora_stats(model)
            w = f" w={st['entry_w'][0]:.4f}/{st['entry_w'][1]:.4f}" if "entry_w" in st else (f" a={st['concat_alpha']:.3f} b={st['concat_beta']:.3f} |BA|={st['concat_BA']:.3f}" if "concat_alpha" in st else "")
            ces = " ".join(f"ce{t}={v:.3f}" for t, v in sorted(agg.items()))
            print(f"[{tag}] S{step}/{total} T={T} loss={tot:.4f} {ces} lr={opt.param_groups[0]['lr']:.2e} gn={gn:.2f}{w} "
                  f"ETA {(total-step)*spi/60:.0f}m ({spi:.2f}s/step)", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "T": T, "loss": tot, "ce": agg, "gn": gn, "lam": lam, **st}) + "\n")

        if step % QUICK_EVERY == 0 and step < total:
            fwd = (lambda x: model(x, num_loops=1, single_pass_lora=True)) if not looped else (lambda x: model(x, num_loops=3))
            res = H.quick_suite(model, fwd, tok, ev, he_n=args.he_n)
            res.update({"kind": "eval", "step": step})
            print(f"[{tag}] EVAL S{step}: {H.fmt(res)}", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(res) + "\n")
            model.train()
        if step % CKPT_EVERY == 0 and step < total:
            save(model, step, tag)

    save(model, total, tag)
    print(f"[{tag}] trained in {(time.time()-t0)/60:.1f} min; ramp opened at {ramp_open_step}", flush=True)

    print(f"\n[{tag}] paired eval vs base", flush=True)
    res = {"arm": arm, "steps": total, "tokens": total * TOK_PER_STEP, "ramp_open_step": ramp_open_step,
           "nll": paired_eval(model, "control" if not looped else "reinject", ev), "gates": H.gate_lora_stats(model)}
    fwd = (lambda x: model(x, num_loops=1, single_pass_lora=True)) if not looped else (lambda x: model(x, num_loops=3))
    he = H.eval_humaneval(fwd, tok, ev["humaneval"], 164 if not args.smoke else 8, log_every=1000)
    res["he164"] = he
    print(f"[{tag}] HE-164: {he['passed']}/{he['total']} ({he['pct']:.1f}%)", flush=True)
    with open(os.path.join(NEWT_HOME, f"{tag}_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {tag}_results.json", flush=True)


def save(model, step, tag):
    keep = set(model.adapter_param_names())
    if any(p.requires_grad for i in range(CORE_START, CORE_END) for p in model.backbone.layers[i].parameters()):
        keep |= {n for n, _ in model.named_parameters() if any(f".layers.{i}." in n for i in range(CORE_START, CORE_END))}
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}
    path = os.path.join(NEWT_HOME, f"{tag}_step{step}.pt")
    torch.save({"step": step, "tag": tag, "preset": PRESET, "state_dict": sd}, path)
    print(f">>> saved {path} ({sum(v.numel() for v in sd.values())/1e6:.1f}M params)", flush=True)


if __name__ == "__main__":
    main()
