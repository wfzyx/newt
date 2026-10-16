# Project Newt — retrofitting recurrent depth onto FrogNano-4B

**Status: reopened 2026-10-17 on a zero-training lane** (see [Training-free loop sweep](#training-free-loop-sweep)).
The training lane is **closed (2026-10-16) as a negative result.** Seven arms across FrogNano-4B and Qwen3.5-0.8B — tanh gates,
zero-init gates + boundary projection, scalar prelude re-injection (2610.00673), concat bridge (2511.07384), with the
core both frozen (LoRA) and fully unfrozen — against compute-matched 1-pass LoRA controls. In every arm the optimizer
routed ≤5% of the signal through the loop; no arm beat its control on all three held-out slices. The literature
(see `bib/`) contains no low-budget depth-loop retrofit that does, and the two retrofits that work (ALoDLM, HybridLink)
change what the loop sees rather than how it is gated.
This README is the hand-off for anyone who wants to pick the idea back up; the section
[What it would take](#what-it-would-take) is the short version.

---

## What this is

An attempt to turn a pretrained dense LM — `microsoft/FrogNano-4B-2609` (Qwen3.5-family hybrid Gated-DeltaNet +
full-attention, 32 layers) — into a looped / recurrent-depth model **without touching the base weights**:

```
Prelude   layers 0–7     frozen base, run once
Core      layers 8–23    run N=3 times; LoRA (r=64) on attn / linear_attn / MLP projections, frozen base
Coda      layers 24–31   frozen base, run once
```

Loop 1 through the core is always the exact base model, so `logits_loop_1` from a single forward pass is a free
reference distribution (used for KL anchoring and for paired per-document evaluation). Loops 2 and 3 are where
the adapters act. The question was whether the extra passes buy anything a 1-pass LoRA of the same size doesn't.

Model code: `model/loopsplit_frognano.py` (`LoopSplitFrogNano`). It supports every variant below via constructor
flags (`gate_mode`, `boundary_adapter`, `entry_mode`, `single_pass_lora`).

## What was tried

| Variant | Loop-entry mechanism | Adapters | Where |
|---|---|---|---|
| Phase A/B/C (steps 0–10k) | layer-23 output fed straight into layer 8; per-layer `tanh(a)` output gates on mixer/MLP, init `a=2.0` (g=0.96) | separate LoRA_2 / LoRA_3 | `train/train_phase_{a,b}.py`, `train/train_resumed_5k_to_10k.py` |
| treatment-500 | same recipe, from base, 500 steps | separate | `train/ablation_500.py --arm treatment` |
| salvage-500 | raw gates zero-init; identity-init full-rank projection + RMSNorm at loop entry; zero-init step embeddings `e_1,e_2`; gate group LR 5e-3 | separate | `--arm salvage` |
| reinject-500 | arXiv:2610.00673 recipe: `z_t = w·p + (1−w)·h_{t−1}`, `w=σ(a)`, `w₀=0.99`, `p` = prelude output; no per-layer gates | **one shared LoRA** | `--arm reinject` |
| control-500 / control-1200 | one pass, LoRA on; the comparison bar (1200 steps ≈ wall-clock of 500 three-loop steps) | one LoRA | `--arm control --steps N` |

All ablation arms: from base init, same data order (50 % general-Python pretrain windows / 31 % docstring
functions / 19 % competitive-programming-as-pure-functions, 13-gram decontaminated vs HumanEval + MBPP), 16
micro-batches of ≤512 tokens per step, AdamW, LoRA LR 5e-5 cosine, clip 1.0, CE-only loss
(`0.25·CE_L2 + 0.75·CE_L3` for looped arms). Data builder: `data/build_phase_d.py`.

## Results

Qwen3.5-0.8B follow-up (unfrozen core, 30M tokens, concat and scalar bridges) is tabulated in
[`bib/README.md`](bib/README.md#what-the-literature-says-about-our-setup); raw files in `results/qwen08_study/`.

Paired per-document cross-entropy delta vs base (negative = better than base), 150 documents per slice,
bootstrap 95 % CI; every CI below excludes zero. HE-40 is greedy pass@1 on the first 40 HumanEval tasks
(n=40 → SE ≈ 7 points; treat as noise). All arms on one A10G. Raw numbers: `results/stage0_ablation/`.

```
Arm                                  GPU-min   TACO (ID)   OOD-pretrain   OOD-func   HE-40
Phase B step 5000 (10× the compute)     —       -0.046       +0.038        +0.038     29
treatment-500  (Phase B recipe)         72      +0.075       +0.026        -0.096     29
salvage-500    (zero-init + proj)       72      -0.024       -0.035        -0.156     32
reinject-500   (2610.00673 recipe)      71      -0.095       -0.057        -0.175     26
control-500    (1-pass LoRA)            30      -0.086       -0.053        -0.173     26
control-1200   (1-pass, compute-matched)72      -0.067       -0.058        -0.180     27
```

Base NLLs: TACO 1.339, OOD-pretrain 1.438, OOD-func 1.192.

Read it as:

1. **The original design (Phases A–C) was never good.** At step 5000 loop 3 beat base only on the in-distribution
   slice and was worse on both OOD slices; loop 2 was worse than base on every slice at every checkpoint
   (`results/stage0_ablation/stage0_results.json`). Phase C's HumanEval collapse (72 % → 45–57 %) was that drift
   crossing the greedy-decoding threshold, not a data problem. The gates sat at 0.963 for 10k steps.
2. **The literature fixes work as described, and then the model declines to use the loop.** reinject-500 starts
   with zero corruption, trains from step 1, keeps loop 1 exact — and ends statistically identical to control-500 on
   all three slices while its mixing weights open only to `w = 0.977 / 0.973` (2–3 % of the signal goes through the
   recurrence). Against the compute-matched control it ties both OOD slices and wins only TACO.
3. **Pre-registered pass bar not met**: a 500-step looped arm had to beat control-1200 on all three slices with
   CIs excluding zero. None did. Nothing multi-pass to ship.

## Why it failed (our reading)

- **Budget, not architecture.** The papers that report gains from retrofitted recurrence
  ([2610.00673](https://arxiv.org/abs/2610.00673), [2511.07384](https://arxiv.org/abs/2511.07384)) do full-weight
  continued training at ≥10⁹ tokens. [2608.11233](https://arxiv.org/abs/2608.11233) installs the mechanism at a
  LoRA budget but on a 0.5B model with an intermediate-state supervision curriculum. We ran ≤8k micro-batches
  (≈4 M loss-bearing tokens) on a frozen 4B base. At that budget the core layers cannot learn to act as an iterative
  refinement operator; the cheapest thing the optimizer can do is use the loop as a slightly noisier 1-pass LoRA.
- **The original gate/entry design was actively harmful** and masked the budget problem for three phases: output
  gates saturated at init, no re-injection of the clean prelude state, gates starved by AdamW normalisation.
  Diagnosed in detail in `bib/README.md` → *What the literature says about our setup*.
- **In-distribution NLL was the only metric for 10k steps.** TACO-test NLL fell monotonically while general-Python
  NLL and HumanEval degraded. Paired OOD deltas against the loop-1 readout (free, since loop 1 is base) would have
  caught this at step 1500. `eval/stage0_loop_diag.py` does this in ~15 min per checkpoint.

## What it would take

If someone wants to continue, the experiments above say the next lever is compute, not another gate design:

1. **Unfreeze the core** (layers 8–23, ~1.8 B params) with the reinject recipe and a shared LoRA removed. Needs
   ≥80 GB (bf16 weights + fp32 AdamW state for the core ≈ 14 GB, plus base, plus activations at 512 tokens × 3 loops);
   an A10G will not do it.
2. **≥100 M tokens** = ~12k steps at 8,192 tokens/step; the papers use 10–100× more. On an A100-80GB expect ~10 h
   at bf16 for the LoRA-only variant, several times that unfrozen.
3. **Add the two curriculum pieces we skipped**: smoothed exit loss (every loop trains, 2610.00673) and
   intermediate-state supervision or a recurrence-count ramp (2608.11233 / 2511.07384).
4. **Keep the bar**: a looped model must beat a 1-pass LoRA at matched *training* compute **and** justify 3× per-token
   inference. [WaveFront decoding](https://arxiv.org/abs/2609.23033) is how looped models pay the latter down; the
   compute-matched Qwen3-4B still beat looped-1.7B in 2610.00673, so expect the margin to be modest even if it exists.

## Training-free loop sweep

A second reading of the literature (bib entries 11–18: 2605.23872, 2602.14759, LoopUS, RYS, …) shows that every
reproducible *zero-training* loop gain uses (i) a **short** window, n ≈ 2–4 layers at 40–60 % depth (sharp cliff at
n ≥ 6), (ii) a **damped** re-application `x ← x + (1/K)(g(x) − x)` or an interpolation toward the base state, and
(iii) K = 2–3. Our seven trained arms satisfied none of these: a 16-layer window with a naive `h ← core(h)` entry.

`model/tf_loop.py` is an inference wrapper over the frozen HF model (no parameters; K=1 is bit-exact base) with
strategies `naive | euler | anchor | uniform` in `block` or `layer` mode. `eval/tf_loop_sweep.py` sweeps 34 windows ×
6 strategies on the existing NLL slices (paired per-doc delta vs base, bootstrap CI), refines the top cells on the full
150-doc slices, and runs HE-164 on the best three. Cost: eval only, ≈ 2.5 h of A10G.

Results go in `results/tf_loop/` when the sweep finishes. Decision rule: a cell counts only if it is ≤ 0 on both OOD
slices with CIs excluding zero *and* does not lose HE-164 by more than noise (≈ ±4). If a cell wins, the next ≤ $20
step is fitting a LoopUS-style decay gate (bib 13) or a TaH-style decider (bib 14) on top of it; if nothing wins,
the project is closed at every budget we can reach.

## Repo layout

```
model/loopsplit_frognano.py   LoopSplitFrogNano + CoreLoopLoRALayer (all variants, flag-selected)
model/presets.py              NEWT_MODEL=frognano|qwen08 → model id + core split
train/study_qwen08.py         Qwen3.5-0.8B arms A/B/C/D (unfrozen core, exit-loss smoothing, T=2→3 ramp, batched)
train/ablation_500.py         the four ablation arms (control / treatment / salvage / reinject), --steps N
train/train_phase_{a,b}.py    Phase A/B as run (historical)
train/train_resumed_5k_to_10k.py   Phase C as run (historical)
train/train_phase_d.py        written, never launched — superseded by the ablations
train/*loss.py                Phase A–D loss functions
data/build_phase_d.py         dataset + eval-slice builder (HF Hub sources; --eval-only, --scale)
eval/phase_d_harness.py       NLL slices, HumanEval-N, MBPP-100, gate/LoRA stats; standalone CLI
eval/stage0_loop_diag.py      per-loop CE vs base with paired bootstrap; loop-count ablation on HE-40
tests/                        Phase A–C eval scripts (historical) + test_step0_parity.py
results/stage0_ablation/      every result JSON and training log from the ablations
model/tf_loop.py              training-free loop wrapper (window, K, strategy, mode) over the frozen HF model
eval/tf_loop_sweep.py         window/strategy sweep → paired NLL deltas, refine, HE-164 on the top cells
bib/                          18 papers (PDF) + annotated README
checkpoints/                  not in git (~10 GB); see below
```

Checkpoints (Phase A/B/C and all ablation arms, ~223 MB each as LoRA+gate state dicts) are on the author's
machine and on the stopped AWS volume `vol-0f22a930cf3171073` (sa-east-1, instance `i-0f3a24c279977f26c`,
`/home/ubuntu/*.pt`). Ask if you need them; nothing in them is worth reproducing.

## Running

```bash
# env: torch 2.6, transformers 5.19, datasets 5.1, evalplus; one 24 GB GPU; NEWT_HOME = where .pt files live
export NEWT_HOME=/path/to/workdir
python -m data.build_phase_d --eval-only                 # 150-doc eval slices + HE-164 + MBPP-100  (~5 min)
python -m data.build_phase_d --scale 0.63 --out $NEWT_HOME/phase_d_data_ablation2.pt   # train data (~10 min)
python -m train.ablation_500 --arm control  --steps 1200 # 1-pass bar          (~75 min A10G)
python -m train.ablation_500 --arm reinject --steps 500  # best looped arm     (~80 min A10G)
python -m eval.stage0_loop_diag                          # per-loop diagnostics on saved checkpoints
python -m eval.phase_d_harness --ckpt X.pt --full        # HE-164 + MBPP-100 + NLL slices
python -m eval.tf_loop_sweep --sweep-docs 80 --refine 10 --he-top 3   # training-free sweep (~2.5 h A10G)
```

Every script prints paired deltas with bootstrap CIs; compare arms on those, not on HE-40.

## References

1. Shapiro. *Retrofitting Recurrent Depth into a Pretrained Language Model.* arXiv:2608.11233 (2026).
2. Marchenko, Bezrukov et al. *Closing the Loop: Practical Training Recipes for Looped Language Models.* arXiv:2610.00673 (2026).
3. McLeish, Li, Kirchenbauer, Kalra et al. *Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence.* arXiv:2511.07384 (2025).
4. Bae, Fisch, Harutyunyan, Ji et al. *Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA.* arXiv:2410.20672 (2024).
5. Geiping, McLeish, Jain, Kirchenbauer et al. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* (Huginn). arXiv:2502.05171 (2025).
6. Hao, Sukhbaatar, Su, Li et al. *Training Large Language Models to Reason in a Continuous Latent Space* (Coconut). arXiv:2412.06769 (2024).
7. Shanghai AI Laboratory. *Intern-S2-Mobius: Decoupled Knowledge and Reasoning.* arXiv:2608.14290 (2026).
8. Ha, Kim. *WaveFront Decoding: Parallelized Self-Speculative Decoding for Looped Language Models.* arXiv:2609.23033 (2026).
9. Fang, Li et al. (Amazon AGI). *ALoDLM: Adaptively Looped Diffusion Language Models.* arXiv:2610.04198 (2026).
10. Zou, Qi, Chen, Bianchi et al. *Hybrid Reasoning Agents.* arXiv:2610.18861 (2026).
11. Li, Liang, Lao, Liu. *Training-Free Looped Transformers.* arXiv:2605.23872 (2026).
12. Lys et al. *Inner Loop Inference for Pretrained Transformers: Unlocking Latent Capabilities Without Training.* arXiv:2602.14759 (2026).
13. *LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models.* arXiv:2605.11011 (2026).
14. Fu et al. *Think-at-Hard: Selective Latent Iterations to Improve Reasoning Language Models.* arXiv:2511.08577 (2025).
15. Guo, Zhu, Ye, Eshraghian. *Allocating Recurrent Compute in Looped Language Models.* arXiv:2608.18230 (2026).
16. Deng, Wang. *Transformers Stop Thinking Too Early, and a Tiny LoRA Fixes It.* arXiv:2609.36585 (2026).
17. Li, Zhou. *Skip a Layer or Loop It? Learning Program-of-Layers in LLMs.* arXiv:2606.06574 (2026).
18. *Layer Duplication in LLMs.* Findings of EMNLP 2025; and Ng, *RYS* (dnhkng.github.io/posts/rys).

Annotated summaries and the mapping from each paper to this repo's design choices: [`bib/README.md`](bib/README.md).

---

Earlier history (an additive 2-pass "Astra" variant with a transition projection, steps 0–20k, evaluated only on
in-distribution NLL) is superseded by the above; its 7.9 GB checkpoint is kept locally only.
