# Newt Bibliography — Retrofitted Recurrent Depth

Papers that bear directly on turning a pretrained dense LM (FrogNano-4B-2609) into a looped / recurrent-depth model
with a frozen base and adapters. PDFs live next to this file. Entries 1–4 are the retrofit blueprints; 5–8 are background.

Status of this repo's own attempts is at the bottom (§ What the literature says about our setup).

---

## 1. Retrofitting Recurrent Depth into a Pretrained Language Model (arXiv:2608.11233)
- **File:** `2608.11233_RetrofittingRecurrentDepth.pdf` — Mark Shapiro, 2026
- **Setup:** Qwen2.5-0.5B-Instruct split at layers 6 and 18 → Prelude (0–5), weight-tied Recurrent Block (6–17), Coda (18–23).
  Two budgets: full-block (180.6M trainable) and **rank-16 LoRA on the looped projections + bridge (6.01M), base frozen**.
  The adapter deltas are weight-tied across loop iterations (one LoRA, not one per loop).
- **Three mechanisms they show are required:**
  1. **Strict loop-1 identity.** `T=1` bypasses every recurrent addition and reproduces the base computation exactly;
     loop-1 general-capability eval is a *preservation* measure, not a different model.
  2. **Split re-entry bridge (Prelude re-injection).** For loops `t > 1` the block input is not `h_t` but
     `u_t = Reentry(p, h_t)` with two separate projections (`W_p · norm(p)` for the Prelude output, `W_s · h_t` for the
     carried state) combined under an **identity-biased gate**. Rationale: `h_t` (block output) lives on a different
     manifold from `p` (block input); feeding it back raw is a distribution shift the block was never trained on.
  3. **The optimizer trap.** A scale/gate folded into a single combined projection "fails silently under both Adam-family
     and orthogonalizing optimizers": Adam's per-element second-moment normalisation cancels constant gradient scales.
     Working fix: gate and prelude projection as **separate tensors in a dedicated optimizer param group with a raised LR**,
     verified at the realised-update level (§5.1).
- Also uses exact intermediate-state supervision (loop `k` state must decode the `k`-th hop) as a curriculum.
- **Relevance:** The closest blueprint to Newt. Same surgery, same frozen-base + LoRA budget, and an explicit diagnosis
  of the two failure modes we hit (loop-entry distribution shift; gates that never move under AdamW).

## 2. Closing the Loop: Practical Training Recipes for Looped Language Models (arXiv:2610.00673)
- **File:** `2610.00673_ClosingTheLoop.pdf` — Marchenko, Bezrukov et al. (DAIMLD), 2026, ICLR 2027 submission
- **Setup:** (a) 1.4B LoopLM from scratch on 310B tokens; (b) **dense→loop conversion of Qwen3-1.7B-Base** with
  continued training (<1% of its pretraining compute — still ≫ our 16M tokens), compared against an *identically
  continued* dense baseline.
- **Reproduces our failure:** "Applying the Qwen3-1.7B-Base decoder recurrently without further changes fails: each later
  step makes the representation worse." A per-step projection with a gate made the gate abandon the middle step.
- **The fix is one scalar:** at the entry of every later step, mix the **original token embedding** `e` with the previous
  recurrent output: `z_t = w·e + (1−w)·h_{t−1}`, `w = σ(a)`, initialised so later steps receive ~99% clean input.
  Plus a *smoothed exit loss* so every step keeps training. With this in place, **step-specific LoRA adapters are
  optional** — they add almost nothing. Looped Qwen beats the continued dense baseline on every benchmark; an
  inference-compute-matched Qwen3-4B-Base still stays ahead.
- **Relevance:** Argues our per-loop LoRA_2/LoRA_3 are the wrong place to spend parameters; the loop boundary is.
  Also fixes the *input* (what enters the block) rather than the *output* (our tanh gates on mixer/MLP outputs).
  Caveat: full-weight continued training, not a frozen-base LoRA retrofit.

## 3. Teaching Pretrained LMs to Think Deeper with Retrofitted Recurrence (arXiv:2511.07384)
- **File:** `2511.07384_RetrofittedRecurrence.pdf` — McLeish, Li, Kirchenbauer, Kalra et al., 2025
- **Summary:** Converts pretrained non-recurrent LMs into Prelude → Recurrent Core → Coda. The core input is the
  **concatenation** of the static Prelude output `p` and the evolving state `h_t`, projected back to `d_model` by a learned
  linear adapter: `h_{t+1} = R(W_proj [h_t ; p])`. A curriculum increases the recurrence count over training; at equal
  compute the converted models beat plain post-training of the original on mathematics.
- **Relevance:** Third independent instance of "always re-inject the Prelude". Concatenation rather than additive
  mixing, so the core never loses the clean early-layer representation — the collapse we saw on TACO and general Python.

## 4. Relaxed Recursive Transformers: Parameter Sharing with Layer-wise LoRA (arXiv:2410.20672)
- **File:** `2410.20672_RelaxedRecursiveTransformers.pdf` — Bae, Fisch, Harutyunyan, Ji et al., 2024
- **Summary:** Ties a block of middle layers (SVD-initialised from the residuals between the original layers) and loops it;
  "relaxed" variants give each loop iteration a depth-wise LoRA so iterations can specialise.
- **Relevance:** Origin of our per-loop LoRA design. But their goal is **parameter compression**, and recovering base
  perplexity after tying needs continual pretraining on billions of tokens — not a 500–5000-step fine-tune. Paper 2
  reports step-specific adapters add ~nothing once the input anchor exists.

---

## Background

## 5. Huginn: Recurrent-Depth Latent Reasoning (arXiv:2502.05171)
- **File:** `2502.05171_Huginn_RecurrentDepth.pdf` — Geiping, McLeish, Jain, Kirchenbauer et al., 2025
- **Summary:** From-scratch 3.5B model (800B tokens) that scales test-time compute by iterating a weight-shared block in
  latent space; reasoning benchmarks improve with recurrence depth.
- **Relevance:** Canonical recurrent-depth reference; the depth-scaling signature our hop-count probe tries to reproduce.
  Trained from scratch with the recurrence present throughout — the opposite regime from a retrofit.

## 6. Coconut: Reasoning in a Continuous Latent Space (arXiv:2412.06769)
- **File:** `2412.06769_Coconut.pdf` — Hao, Sukhbaatar, Su, Li et al., 2024
- **Summary:** Feeds the last hidden state back as the next input embedding instead of decoding a token; multi-stage
  curriculum gradually replaces language CoT with continuous thoughts.
- **Relevance:** Same hidden-state-as-input mechanism as our loop boundary. Its dependence on a curriculum is a warning
  against single-stage training from a saturated gate.

## 7. Intern-S2-Mobius: Decoupled Knowledge and Reasoning (arXiv:2608.14290)
- **File:** `2608.14290_InternS2Mobius.pdf` — Shanghai AI Laboratory, 2026
- **Summary:** Shared FFN memory queried iteratively by attention "reasoners"; Intern-S2-Mobius is continually pretrained
  from **Qwen3.5-35B** (FrogNano's family) and matches downstream scores at ~4× inference speedup. §4.1.2 surveys looped LMs.
- **Relevance:** Evidence the Qwen3.5 lineage tolerates a looped retrofit — via large-scale continual pretraining, not LoRA.

## 8. WaveFront Decoding: Parallelized Self-Speculative Decoding for Looped LMs (arXiv:2609.23033)
- **File:** `2609.23033_WaveFrontDecoding.pdf` — Ha, Kim (SNU), 2026
- **Summary:** Batches tokens at different sequence positions and recurrence depths into one forward pass (diagonal
  pipeline), removing the `T×` weight-streaming cost of autoregressive generation with a looped block.
- **Relevance:** If a multi-pass Newt ever ships, this is how the 3× per-token inference cost gets paid down. Also relevant
  to the compute-matched comparison bar (a looped model must beat a 1-pass model at equal *inference* cost, not just
  equal training time).

---

## What the literature says about our setup

| Newt (Phases A–C, ablation treatment) | What papers 1–3 do instead |
|---|---|
| Layer-23 output fed straight back into layer 8 | Re-inject the Prelude (layer-7) output at every loop entry: split bridge (1), scalar mix with token embedding (2), concat + projection (3) |
| Output gates `g = tanh(a)` on mixer/MLP, init `a = 2.0` → 96 % loop state from step 1 | Gate the **input**, init ≈ 99 % clean state / 1 % looped state; loop 1 is bit-exact base |
| Gates in the same AdamW group as LoRA (or at 0.1× LR); never moved in 10k steps | Gate + bridge as separate tensors in their own param group with raised LR (1) |
| Separate LoRA_2 / LoRA_3 carrying all of the alignment burden | One shared LoRA (or frozen base) + a full-rank boundary projection / scalar mixer; step-specific adapters add ≈ 0 (2) |
| 16M-token budget, LoRA only | 2 and 3 use full-weight continued training at ≥ 10⁹ tokens; 1 shows the mechanism *installs* at a 6M-param LoRA budget on a 0.5B model, with an intermediate-state curriculum |

Measured in this repo (`results/stage0_ablation/`; paired per-doc CE delta vs base, 150 docs/slice, bootstrap 95% CI;
negative = better than base; all arms trained from base on the same 50/31/19 mix):

| Arm | GPU-min | TACO (ID) | OOD-pretrain | OOD-func | HE-40 |
|---|---|---|---|---|---|
| B-5000 (Phase B, tanh gates a=2.0, per-loop LoRA) | — | −0.046 | **+0.038** | **+0.038** | 29 |
| treatment-500 (same recipe from base) | 72 | **+0.075** | **+0.026** | −0.096 | 29 |
| salvage-500 (zero-init raw gates + boundary proj + step emb) | 72 | −0.024 | −0.035 | −0.156 | 32 |
| reinject-500 (paper-2 recipe: `w·p + (1−w)·h`, w₀=0.99, shared LoRA) | 71 | −0.095 | −0.057 | −0.175 | 26 |
| control-500 (1-pass LoRA) | 30 | −0.086 | −0.053 | −0.173 | 26 |
| control-1200 (1-pass LoRA, compute-matched to a 500-step 3-loop run) | 72 | −0.067 | −0.058 | −0.180 | 27 |

- Stage 0: at every Phase B/C checkpoint loop 3 beats base only on TACO and is worse on both OOD slices; loop 2 is
  worse than base everywhere; gates sat at 0.963 throughout.
- The paper-2 recipe behaves as described (no corruption at init, gradient live from step 1, loop 1 exact base) and
  ends statistically identical to the 1-pass control at equal *steps*; it ties the compute-matched control on both OOD
  slices and wins only TACO. The learned mix weights opened to `w = 0.977 / 0.973` — the model routes ~2–3 % through the
  recurrence and leaves the rest as a 1-pass LoRA.
- Conclusion at this budget (≤ 8k micro-batches, frozen base, LoRA): none of the three loop-boundary designs produces a
  multi-pass model that beats a 1-pass LoRA at matched compute. The papers that do report gains use ≥ 10⁹ tokens of
  full-weight continued training (2, 3) or an intermediate-state curriculum on a 0.5B model (1).
