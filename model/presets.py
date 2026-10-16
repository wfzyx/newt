"""
Model presets. Select with NEWT_MODEL=frognano|qwen08 (default frognano).

Splits keep each loop ending on a full-attention layer (layer pattern [lin,lin,lin,full] in both models).
qwen08 uses Shapiro's 6/18 split (arXiv:2608.11233) proportionally: prelude 0-5, core 6-17, coda 18-23.
"""
import os

PRESETS = {
    "frognano": {"model_id": "microsoft/FrogNano-4B-2609", "core_start": 8, "core_end": 24, "lora_rank": 64},
    "qwen08":   {"model_id": "Qwen/Qwen3.5-0.8B",          "core_start": 6, "core_end": 18, "lora_rank": 64},
}

NAME = os.environ.get("NEWT_MODEL", "frognano")
assert NAME in PRESETS, f"NEWT_MODEL must be one of {list(PRESETS)}"
P = PRESETS[NAME]
MODEL_ID = P["model_id"]
CORE_START = P["core_start"]
CORE_END = P["core_end"]
LORA_RANK = P["lora_rank"]
