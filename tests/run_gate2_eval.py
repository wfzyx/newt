import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.loopsplit_frognano import LoopSplitFrogNano

MODEL_ID = "microsoft/FrogNano-4B-2609"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
CHECKPOINT_PATH = "/home/ubuntu/loopsplit_pilot_step300.pt"

PROMPTS = [
    # 1. Standard HumanEval docstring completion
    "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
    # 2. Logic & algorithmic task
    "def separate_paren_groups(paren_string: str) -> list[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced, each group as a string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
    # 3. Tool / agent edit behavior check
    "You are a coding assistant. Write a python function `fix_docstring` that replaces outdated docstrings.\n```python\n",
]

def generate_autoregressive(model, tokenizer, prompt, max_new_tokens=128, num_loops=3):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    curr_ids = input_ids
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(curr_ids, num_loops=num_loops)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_token], dim=-1)
            if next_token.item() in [tokenizer.eos_token_id, 151643, 151645]: # common EOS
                break
                
    generated = tokenizer.decode(curr_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return generated

def run_gate2_eval():
    print(f"=== Starting Gate 2: Generation Sanity & Syntax Check ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    base_model.eval()
    
    # 1. Initialize LoopSplit model
    model = LoopSplitFrogNano(base_model, core_start=8, core_end=24, lora_rank=64).to(DEVICE)
    
    # 2. Load trained checkpoint
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.g2_mixer.data.copy_(ckpt["g2_mixer"])
    model.g2_mlp.data.copy_(ckpt["g2_mlp"])
    model.g3_mixer.data.copy_(ckpt["g3_mixer"])
    model.g3_mlp.data.copy_(ckpt["g3_mlp"])
    
    # Load LoRA weights
    model_state = model.state_dict()
    for k, v in ckpt["state_dict"].items():
        if k in model_state:
            model_state[k].copy_(v)
    model.eval()
    print("Checkpoint step 300 loaded successfully.\n")
    
    for idx, prompt in enumerate(PROMPTS, 1):
        print(f"--- Prompt {idx} ---")
        print(prompt[:120].strip() + "...\n")
        
        # Test 1-loop (Base pass) vs 3-loops (Recursive reasoning)
        gen_1loop = generate_autoregressive(model, tokenizer, prompt, max_new_tokens=96, num_loops=1)
        gen_3loop = generate_autoregressive(model, tokenizer, prompt, max_new_tokens=96, num_loops=3)
        
        print(f"[Loop 1 Output (Base)]:\n{gen_1loop}\n")
        print(f"[Loop 3 Output (Recursive Core)]:\n{gen_3loop}\n")
        print("=" * 60)

if __name__ == "__main__":
    run_gate2_eval()
