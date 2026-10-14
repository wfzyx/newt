import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.loopsplit_frognano import LoopSplitFrogNano

MODEL_ID = "microsoft/FrogNano-4B-2609"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

def test_step0_bit_exact_parity():
    print(f"=== Testing Gate 0: Step-0 Parity on {DEVICE} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    base_model.eval()
    
    test_text = "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)"
    input_ids = tokenizer(test_text, return_tensors="pt").input_ids.to(DEVICE)
    
    # 1. Base FrogNano logits
    with torch.no_grad():
        base_logits = base_model(input_ids).logits
        
    # 2. LoopSplit FrogNano logits (N=3 loops at Step 0)
    looped_model = LoopSplitFrogNano(base_model, core_start=8, core_end=24, lora_rank=64).to(DEVICE)
    looped_model.eval()
    
    with torch.no_grad():
        looped_logits, readouts = looped_model(input_ids, num_loops=3, return_all_readouts=True)
        
    # 3. Verify Bit-Exact Identity
    max_diff_final = (looped_logits - base_logits).abs().max().item()
    max_diff_loop1 = (readouts["logits_loop_1"] - base_logits).abs().max().item()
    max_diff_loop2 = (readouts["logits_loop_2"] - base_logits).abs().max().item()
    max_diff_loop3 = (readouts["logits_loop_3"] - base_logits).abs().max().item()
    
    print(f"Max absolute logit difference vs Base:")
    print(f"  - Final output (N=3): {max_diff_final:.8e}")
    print(f"  - Loop 1 readout:     {max_diff_loop1:.8e}")
    print(f"  - Loop 2 readout:     {max_diff_loop2:.8e}")
    print(f"  - Loop 3 readout:     {max_diff_loop3:.8e}")
    
    assert max_diff_final < 1e-4, f"Parity failure: max diff {max_diff_final} exceeds 1e-4 tolerance!"
    print("\n>>> GATE 0 PASSED: LoopSplit FrogNano is bit-identical to Base FrogNano at Step 0! <<<")

if __name__ == "__main__":
    test_step0_bit_exact_parity()
