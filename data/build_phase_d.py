"""
Phase D dataset builder.

Produces one file (default: $NEWT_HOME/phase_d_data.pt) containing:
  train:
    stream1  (general Python replay, pretrain-style 512-token windows, loss on all tokens)
    stream2  (clean docstring functions: prompt = imports+signature+docstring masked, loss on body + EOS)
    stream3  (competitive reasoning rewritten as pure functions: prompt masked, loss on code; no stdin)
  eval:
    taco_test     (150 docs, ID NLL — identical construction to run_checkpoint_trend_eval.py)
    ood_pretrain  (150 docs, held-out general Python, same format as stream1)
    ood_func      (150 docs, held-out docstring functions, same format as stream2; NLL on body only)
    humaneval     (164 tasks: prompt / entry_point / test)
    mbpp100       (100 sanitized test tasks: prompt / code / test_list)

Every train sample: {"input_ids": LongTensor[T], "loss_mask": BoolTensor[T], "stream": int}
Decontamination: 13-gram (whitespace tokens) collision against HumanEval-164 prompts+canonical
solutions and MBPP-sanitized (all splits) prompts+code.

Dataset sources are HF Hub ids; each stream has a primary and a fallback. Run once on the box
with network access; training reads only the .pt.
"""
import os
import re
import json
import random
import argparse
from typing import Iterable, Iterator, Optional

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from evalplus.data import get_human_eval_plus

from model.presets import MODEL_ID
MAX_LEN = 512
SEED = 20261015
NEWT_HOME = os.environ.get("NEWT_HOME", "/home/ubuntu")

# 8 : 5 : 3 micro-batches per optimizer step, 2000 steps, no sample repetition + 5% slack
STEPS = 2000
PER_STEP = {1: 8, 2: 5, 3: 3}
TARGET = {k: int(v * STEPS * 1.05) for k, v in PER_STEP.items()}  # 16800 / 10500 / 6300
EVAL_DOCS = 150

STDIN_RE = re.compile(r"\binput\s*\(|sys\.stdin|stdin\.|open\(0\)|readline\(|\bprint\s*\(")


# --------------------------------------------------------------------------- decontam
def ngrams13(text: str) -> Iterator[str]:
    toks = text.split()
    for i in range(len(toks) - 12):
        yield " ".join(toks[i:i + 13])


class Decontaminator:
    def __init__(self, texts: Iterable[str]):
        self.ngrams = set()
        for t in texts:
            self.ngrams.update(ngrams13(t))
        self.dropped = 0

    def contaminated(self, text: str) -> bool:
        for g in ngrams13(text):
            if g in self.ngrams:
                self.dropped += 1
                return True
        return False


def build_decontaminator() -> Decontaminator:
    texts = []
    for p in get_human_eval_plus().values():
        texts.append(p["prompt"] + p.get("canonical_solution", ""))
    for cfg in ("sanitized", "full"):
        for split in ("train", "validation", "test", "prompt"):
            try:
                ds = load_dataset("google-research-datasets/mbpp", cfg, split=split)
            except Exception:
                continue
            for it in ds:
                texts.append((it.get("prompt") or it.get("text") or "") + "\n" + it.get("code", ""))
    d = Decontaminator(texts)
    print(f"  decontam: {len(d.ngrams)} 13-grams indexed from HumanEval-164 + MBPP", flush=True)
    return d


# --------------------------------------------------------------------------- helpers
def make_sample(tok, prompt: str, target: str, stream: int, min_target_toks: int = 16) -> Optional[dict]:
    """prompt tokens masked out of the loss, target tokens in the loss. Truncates to MAX_LEN."""
    p_ids = tok.encode(prompt, add_special_tokens=False) if prompt else []
    t_ids = tok.encode(target, add_special_tokens=False)
    if len(p_ids) >= MAX_LEN - min_target_toks:
        return None
    ids = (p_ids + t_ids)[:MAX_LEN]
    n_target = len(ids) - len(p_ids)
    if n_target < min_target_toks:
        return None
    mask = torch.zeros(len(ids), dtype=torch.bool)
    mask[len(p_ids):] = True
    return {"input_ids": torch.tensor(ids, dtype=torch.long), "loss_mask": mask, "stream": stream}


def first_of(ds_specs):
    """Try a list of (repo, kwargs) specs; return the first that loads."""
    last = None
    for repo, kw in ds_specs:
        try:
            ds = load_dataset(repo, **kw)
            print(f"  loaded {repo} {kw}", flush=True)
            return ds
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  WARN: {repo} failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
    raise RuntimeError(f"no dataset spec loadable: {last}")


# --------------------------------------------------------------------------- stream 1
def iter_python_windows(tok, ds, text_key_candidates=("content", "code", "text")) -> Iterator[list]:
    """Pretrain-style: concatenate docs with EOS, emit non-overlapping 512-token windows."""
    buf: list[int] = []
    eos = tok.eos_token_id
    for item in ds:
        text = next((item[k] for k in text_key_candidates if k in item and item[k]), None)
        if not text:
            continue
        lang = item.get("language") or item.get("lang") or "python"
        if isinstance(lang, str) and lang.lower() != "python":
            continue
        buf.extend(tok.encode(text, add_special_tokens=False) + [eos])
        while len(buf) >= MAX_LEN:
            yield buf[:MAX_LEN]
            buf = buf[MAX_LEN:]


def build_stream1(tok, decon, n_train: int, n_eval: int):
    print("[stream 1] general Python replay", flush=True)
    train_ds = first_of([
        ("codeparrot/codeparrot-clean-train", {"split": "train", "streaming": True}),
        ("bigcode/the-stack-smol", {"data_dir": "data/python", "split": "train", "streaming": True}),
    ])
    eval_ds = first_of([
        ("codeparrot/codeparrot-clean-valid", {"split": "train", "streaming": True}),
        ("bigcode/the-stack-smol", {"data_dir": "data/python", "split": "train", "streaming": True}),
    ])
    train, evals = [], []
    for ids in (iter_python_windows(tok, train_ds) if n_train > 0 else ()):
        if decon.contaminated(tok.decode(ids)):
            continue
        train.append({"input_ids": torch.tensor(ids), "loss_mask": torch.ones(len(ids), dtype=torch.bool), "stream": 1})
        if len(train) >= n_train:
            break
    # held-out eval windows come from the *valid* shard; skip the first 2000 windows if sharing a source
    skip = 2000 if eval_ds is train_ds else 0
    for i, ids in enumerate(iter_python_windows(tok, eval_ds)):
        if i < skip:
            continue
        evals.append(torch.tensor(ids))
        if len(evals) >= n_eval:
            break
    print(f"  stream1: {len(train)} train windows, {len(evals)} eval windows", flush=True)
    return train, evals


# --------------------------------------------------------------------------- stream 2
IMPORT_RE = re.compile(r"^(?:from\s+\S+\s+import\s+.+|import\s+.+)$", re.M)


def split_function(code: str):
    """
    Returns (prompt, body) for a function whose first statement is a docstring, else None.
      prompt = 'def sig(...):' + newline + indented triple-quoted docstring + newline
      body   = rest of function
    """
    m = re.match(r"\s*(def\s+\w+\s*\(.*?\)\s*(?:->\s*[^:]+)?:\s*\n)", code, re.S)
    if not m:
        return None
    header = m.group(1)
    rest = code[m.end():]
    dm = re.match(r'(\s*(?:"""|\'\'\')[\s\S]*?(?:"""|\'\'\')\s*\n)', rest)
    if not dm:
        return None
    docstring = dm.group(1)
    body = rest[dm.end():]
    if len(body.strip()) < 20 or "\n" not in body.strip():
        return None
    if body.strip().startswith("pass") or "NotImplemented" in body:
        return None
    return header + docstring, body


def build_stream2(tok, decon, n_train: int, n_eval: int):
    print("[stream 2] clean docstring functions", flush=True)
    ds = first_of([
        ("Nan-Do/code-search-net-python", {"split": "train"}),
        ("code-search-net/code_search_net", {"name": "python", "split": "train"}),
    ])
    train, evals = [], []
    eos = tok.eos_token or "<|endoftext|>"
    rng = random.Random(SEED)
    for item in ds:
        code = item.get("code") or item.get("func_code_string") or item.get("original_string") or ""
        if not code or len(code) > 6000:
            continue
        partition = item.get("partition", "train")
        sp = split_function(code)
        if sp is None:
            continue
        prompt, body = sp
        # keep stream-2 strict: no I/O, no class methods (self), no decorators
        if STDIN_RE.search(body) or "self" in prompt.split("(")[1].split(")")[0]:
            continue
        full = prompt + body
        if decon.contaminated(full):
            continue
        # Optional import header: 50% of samples carry the file-level imports if the dataset has them
        imports = ""
        src_file = item.get("original_string") if item.get("original_string") != code else None
        if src_file and rng.random() < 0.5:
            found = IMPORT_RE.findall(src_file)
            if found:
                imports = "\n".join(found[:6]) + "\n\n"
        s = make_sample(tok, imports + prompt, body.rstrip() + "\n\n" + eos, stream=2)
        if s is None:
            continue
        if partition in ("valid", "validation", "test") and len(evals) < n_eval:
            evals.append(s)
        elif partition == "train" and len(train) < n_train:
            train.append(s)
        if len(train) >= n_train and len(evals) >= n_eval:
            break
    if len(evals) < n_eval:  # dataset without partition column: carve eval off the tail
        k = n_eval - len(evals)
        evals.extend(train[-k:])
        train = train[:-k]
    print(f"  stream2: {len(train)} train, {len(evals)} eval", flush=True)
    return train, evals


# --------------------------------------------------------------------------- stream 3
def parse_solutions(raw):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    return raw if isinstance(raw, list) else []


def is_pure_function_solution(code: str) -> bool:
    return ("def " in code) and not STDIN_RE.search(code) and ("if __name__" not in code)


def build_stream3(tok, decon, n_train: int):
    """
    TACO + code_contests restricted to solutions that are already pure functions
    (contain `def`, no stdin/print/__main__). We do not have an LLM-rewrite step here, so
    'rewritten strictly into clean functions' is implemented as a hard filter; the call-based
    TACO subset (fn_name present) is preferred.
    """
    print("[stream 3] competitive reasoning as pure functions", flush=True)
    train = []
    taco = load_dataset("parquet", data_files="hf://datasets/BAAI/TACO/ALL/train-*.parquet", split="train")
    call_based, other = [], []
    for item in taco:
        sols = [s for s in parse_solutions(item.get("solutions", "[]")) if is_pure_function_solution(s.strip())]
        if not sols:
            continue
        io = item.get("input_output") or ""
        is_call = '"fn_name"' in io if isinstance(io, str) else bool(io.get("fn_name"))
        q = (item.get("question") or "").strip()
        starter = (item.get("starter_code") or "").strip()
        sol = sols[0].strip()
        if len(sol) < 40:
            continue
        prompt = f"# Problem\n{q}\n\n# Solution\n"
        if starter and starter.split("\n")[0] not in sol:
            prompt += starter.split("\n")[0] + "\n"
        (call_based if is_call else other).append((prompt, sol))
    print(f"  TACO pure-function candidates: call-based={len(call_based)} other={len(other)}", flush=True)

    contests = []
    try:
        cc = load_dataset("deepmind/code_contests", split="train", streaming=True)
        for item in cc:
            sols = item.get("solutions", {})
            for s, lang in zip(sols.get("solution", []), sols.get("language", [])):
                if lang == 3 and is_pure_function_solution(s) and len(s.strip()) >= 40:
                    contests.append((f"# Problem\n{item['description'].strip()}\n\n# Solution\n", s.strip()))
                    break
            if len(contests) >= n_train // 4:
                break
    except Exception as e:  # noqa: BLE001
        print(f"  WARN code_contests unavailable: {e}", flush=True)
    print(f"  code_contests pure-function candidates: {len(contests)}", flush=True)

    rng = random.Random(SEED + 3)
    pool = call_based + contests
    rng.shuffle(pool)
    rng.shuffle(other)
    pool += other  # non-call-based TACO only as backfill
    for prompt, sol in pool:
        if decon.contaminated(prompt + sol):
            continue
        s = make_sample(tok, prompt, sol + "\n", stream=3, min_target_toks=24)
        if s is None:
            continue
        train.append(s)
        if len(train) >= n_train:
            break
    print(f"  stream3: {len(train)} train", flush=True)
    return train


# --------------------------------------------------------------------------- eval sets
def build_taco_test(tok, n: int):
    """Identical construction to tests/run_checkpoint_trend_eval.py (ID NLL continuity with Phase C numbers)."""
    ds = load_dataset("parquet", data_files="hf://datasets/BAAI/TACO/ALL/test-*.parquet", split="train")
    out = []
    for item in ds:
        sols = parse_solutions(item.get("solutions", []))
        if not sols or len(sols[0].strip()) < 40:
            continue
        text = f"# Problem\n{item.get('question', '').strip()}\n\n# Solution\n{sols[0].strip()}"
        toks = tok.encode(text, max_length=MAX_LEN, truncation=True)
        if len(toks) >= 128:
            out.append(torch.tensor(toks, dtype=torch.long))
        if len(out) >= n:
            break
    print(f"  taco_test: {len(out)} docs", flush=True)
    return out


def build_mbpp100():
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    items = []
    for it in ds:
        items.append({"task_id": it["task_id"], "prompt": it["prompt"], "code": it["code"],
                      "test_list": list(it["test_list"]), "test_imports": list(it.get("test_imports", []))})
        if len(items) >= 100:
            break
    return items


def build_humaneval():
    probs = get_human_eval_plus()
    return [{"task_id": k, "prompt": v["prompt"], "entry_point": v["entry_point"], "test": v["test"]}
            for k, v in probs.items()]


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(NEWT_HOME, "phase_d_data.pt"))
    ap.add_argument("--scale", type=float, default=1.0, help="shrink targets for a smoke build")
    ap.add_argument("--eval-only", action="store_true", help="build only the eval slices (Stage 0)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    decon = build_decontaminator()
    tgt = {k: max(32, int(v * args.scale)) for k, v in TARGET.items()}
    n_eval = max(8, int(EVAL_DOCS * args.scale))

    if args.eval_only:
        tgt = {k: 0 for k in tgt}
    s1_train, s1_eval = build_stream1(tok, decon, tgt[1], n_eval)
    s2_train, s2_eval = build_stream2(tok, decon, tgt[2], n_eval)
    s3_train = [] if args.eval_only else build_stream3(tok, decon, tgt[3])
    taco_test = build_taco_test(tok, n_eval)

    for k, lst, need in ((1, s1_train, PER_STEP[1] * STEPS), (2, s2_train, PER_STEP[2] * STEPS), (3, s3_train, PER_STEP[3] * STEPS)):
        if args.scale >= 1.0 and not args.eval_only and len(lst) < need:
            print(f"  WARN stream{k}: {len(lst)} < {need}; training would repeat samples", flush=True)

    payload = {
        "meta": {"model": MODEL_ID, "max_len": MAX_LEN, "seed": SEED, "per_step": PER_STEP,
                 "decontam_dropped": decon.dropped, "counts": {1: len(s1_train), 2: len(s2_train), 3: len(s3_train)}},
        "train": {1: s1_train, 2: s2_train, 3: s3_train},
        "eval": {
            "taco_test": taco_test,
            "ood_pretrain": s1_eval,
            "ood_func": s2_eval,
            "humaneval": build_humaneval(),
            "mbpp100": build_mbpp100(),
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(payload, args.out)
    tot = sum(len(v) for v in payload["train"].values())
    print(f"\n=== Phase D data written: {args.out} ===")
    for k in (1, 2, 3):
        n = len(payload["train"][k])
        print(f"  stream{k}: {n} ({n / max(tot, 1) * 100:.1f}%)")
    print(f"  eval: taco={len(taco_test)} ood_pretrain={len(s1_eval)} ood_func={len(s2_eval)} "
          f"he={len(payload['eval']['humaneval'])} mbpp={len(payload['eval']['mbpp100'])}")
    print(f"  decontam collisions dropped: {decon.dropped}")


if __name__ == "__main__":
    main()
