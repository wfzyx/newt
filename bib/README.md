# Newt Bibliography — Retrofitted Recurrent Depth

Papers that bear directly on turning a pretrained dense LM (FrogNano-4B-2609) into a looped / recurrent-depth model
with a frozen base and adapters. PDFs live next to this file. Entries 1–4 are the retrofit blueprints; 5–8 are background;
9–10 are the two recurrence designs that *do* beat their own base after a retrofit, and why they differ from ours.

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

## Retrofits that work (and what they spend)

## 9. ALoDLM: Adaptively Looped Diffusion Language Models (arXiv:2610.04198)
- **File:** `2610.04198_ALoDLM.pdf` — Fang, Li et al. (Amazon AGI), 2026. Code: `amazon-science/ALoDLM`; weights CC BY-NC.
- **Setup:** Qwen3-1.7B / Qwen3-8B converted to **masked-diffusion** LMs with a looped core. 8B split: Prelude [0,10),
  shared Recurrent Core [10,26), Coda [26,36) — the same surgery as Newt. Max depth K=4. **Direct SFT, full weights,
  5B tokens**, no continued pretraining.
- **Mechanism:** inside each denoising step the core runs up to K times; after each pass the LM head commits tokens whose
  entropy ≤ τ and feeds them back as *token embeddings*, while unresolved positions carry their latent state into the
  next pass. A learned exit gate (trained as a latent-variable NELBO, truncated-geometric prior) halts per token.
- **Results:** average over 11 benchmarks 65.5 vs 63.8 (1.7B) and **80.3 vs 78.5 (8B)** against the AR Qwen3 base;
  HumanEval 87.8 vs 85.4. ≈2.7× the throughput of vLLM Qwen3-8B on GSM8K at matched accuracy (their engine, one B200).
- **Why it does not contradict Newt:** the loop is load-bearing by construction — pass *t+1* sees newly committed tokens
  pass *t* did not, so "ignore the loop" is not a reachable minimum. Per-token adaptive depth is what turns extra compute
  into *less* average compute. And it is full-weight SFT at ~150× our token budget. The only retrofit in this list that
  improves quality and speed together; the cost is leaving the AR decoding paradigm (needs their decoder).

## 10. Hybrid Reasoning Agents / HybridLink (arXiv:2610.18861)
- **File:** `2610.18861_HybridAgent.pdf` — Zou, Qi, Chen, Bianchi et al. (Together AI et al.), 2026. Code: `togethercomputer/HybridAgent`.
- **Setup:** recurrence over *time*, not depth. A ~10M-param residual adapter `e_{t+1} = h_t + W₂σ(W₁h_t)` maps the
  **last-layer** hidden state back to the **input embedding** of the next position; between `<latent>…</latent>` the model
  generates without the LM head (Coconut with a learned bridge and an exit anchor). Normal AR + tools outside the span.
- **Training:** HybridSFT stage 1 = text SFT of the backbone (anchors); stage 2 = **backbone frozen, train only HybridLink**
  by cosine+MSE regression to target embeddings; then **HybridRL with full-parameter backbone** on multi-node H100s
  (App. H.3). No GPU-hour figure published.
- **Results (Qwen3-4B-Base, Table 8):** text-SFT 77.4 MATH500 / 19.0 AIME25 → HybridAgent-SFT 76.8 / 22.5 (≈ tie);
  text-RL (ARPO) 81.8 / 31.3 → **HybridAgent-RL 82.2 / 35.2**. Headline +10.4% is vs other agent frameworks. Real win is
  efficiency: 35–55% fewer tokens, 1.8–2.6× faster. Ablation (Table 6, 14B): identity bridge 71.4 MATH500, 2-layer
  residual 88.2.
- **Why it does not hit Newt's wall:** last-layer → embedding is the model's own `lm_head∘embed` round-trip, a distribution
  it already handles every token (identity bridge already works); and a latent step *is* the next position, so there is
  no gate to collapse. The cheap frozen-backbone stage installs the mechanism; the accuracy comes from RL.

**Pattern across 9–10 vs 1–3 and our arms:** the retrofits that pay off either give the loop information the previous pass
did not have (ALoDLM: committed tokens; HybridLink: it is the next token), or spend ≥10⁹ full-weight tokens, or both.
A depth loop over the same input with a frozen base and ≤10⁸ tokens has no published success, and seven arms here
reproduced that absence.

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

Follow-up on **Qwen3.5-0.8B** (same family, 24 layers, core 6–17; 30M tokens = 3,662 steps; `results/qwen08_study/`).
Deltas vs the *frozen original* base; A's own eval gives base NLLs TACO 1.617 / pretrain 1.722 / func 1.496.

| Arm | Trainable | GPU-min | TACO | OOD-pretrain | OOD-func | HE-164 |
|---|---|---|---|---|---|---|
| A  1-pass LoRA r=64, frozen base | 18.7M | 166 | −0.030 | −0.055 | −0.181 | 36 (22.0%) |
| C  3-loop reinject (w₀=0.99), **core unfrozen** | 286M | 503 | −0.057 | −0.068 | −0.187 | 30 (18.3%) |
| D  3-loop concat bridge (2511.07384), core unfrozen | 288M | killed @ S950 | — | — | — | — |

- C's loop-3 vs *its own* loop-1 (isolates the recurrence): TACO −0.002 [CI incl. 0], pretrain −0.005, func −0.020.
  Mix weights opened to w = 0.959 / 0.955 and flattened; ce₂ ≡ ce₃ throughout. Almost all of C's margin over A is in its
  1-pass column (unfreezing 268M more params), not in the loop.
- D's bridge spent its capacity re-encoding the prelude (β 1.0→0.40, |BA|↑) and shrank the carried-state gain
  (α 0.05→0.02). Same collapse as C by a different route; stopped at 26%.
