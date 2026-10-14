import json
from evalplus.data import get_human_eval_plus
from evalplus.evaluate import check_correctness

INPUT_FILE = "/home/ubuntu/humaneval_eval_results/loopsplit_humaneval_loop3_n30.jsonl"

def score_subset():
    problems = get_human_eval_plus()
    
    with open(INPUT_FILE, "r") as f:
        samples = [json.loads(line) for line in f]
        
    print(f"Scoring {len(samples)} HumanEval completions...")
    
    base_passed = 0
    plus_passed = 0
    total = len(samples)
    
    for item in samples:
        task_id = item["task_id"]
        completion = item["completion"]
        problem = problems[task_id]
        
        # Check base and plus tests
        res_base = check_correctness(dataset="humaneval", completion_id=0, problem=problem, solution=completion, expected_output=problem.get("base_input_output"), base_only=True, fast_check=False)
        res_plus = check_correctness(dataset="humaneval", completion_id=0, problem=problem, solution=completion, expected_output=problem.get("plus_input_output"), base_only=False, fast_check=False)
        
        is_base = res_base[0] == "pass" if isinstance(res_base, (list, tuple)) else (res_base == "pass" or res_base.get("base", [None])[0] == "pass")
        is_plus = res_plus[0] == "pass" if isinstance(res_plus, (list, tuple)) else (res_plus == "pass" or res_plus.get("plus", [None])[0] == "pass")
        
        if is_base:
            base_passed += 1
        if is_plus:
            plus_passed += 1
            
        status = "PASS (Base+Plus)" if (is_base and is_plus) else ("PASS (Base only)" if is_base else "FAIL")
        print(f"[{task_id}] -> {status}")
        if not is_base:
            print(f"   Completion: {repr(completion[:100])}")

    print("\n" + "=" * 50)
    print(f"HumanEval Base Pass@1 (30 tasks): {base_passed}/{total} ({base_passed/total*100:.1f}%)")
    print(f"HumanEval+  Plus Pass@1 (30 tasks): {plus_passed}/{total} ({plus_passed/total*100:.1f}%)")
    print("=" * 50)

if __name__ == "__main__":
    score_subset()
