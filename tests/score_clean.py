import json
from evalplus.data import get_human_eval_plus

SAMPLE_FILE = "/home/ubuntu/humaneval_eval_results/loopsplit_humaneval_loop3_n30.jsonl"

def score_standard():
    problems = get_human_eval_plus()
    with open(SAMPLE_FILE) as f:
        samples = [json.loads(line) for line in f]

    passed = 0
    total = len(samples)

    print(f"=== Evaluating Standard HumanEval ({total} tasks) ===")
    for item in samples:
        tid = item["task_id"]
        prob = problems[tid]
        entry = prob["entry_point"]
        
        # Build clean test script
        test_script = prob["prompt"] + item["completion"] + "\n" + prob["test"] + f"\ncheck({entry})\n"
        
        try:
            exec(test_script, {})
            passed += 1
            print(f"[{tid}] PASS")
        except Exception as e:
            print(f"[{tid}] FAIL -> {type(e).__name__}: {str(e)[:60]}")

    acc = (passed / total) * 100
    print("\n" + "=" * 50)
    print(f"LoopSplit Step 5000 HumanEval Pass@1: {passed}/{total} ({acc:.1f}%)")
    print("=" * 50)

if __name__ == "__main__":
    score_standard()
