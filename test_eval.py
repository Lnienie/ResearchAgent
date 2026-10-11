"""
Evaluation script for test_data.jsonl.
Usage:
    python3 test_eval.py                          # run all 10 questions
    python3 test_eval.py 0 5                      # run questions id 0-4
    python3 test_eval.py 0 10 my_output.jsonl     # custom output file
"""
import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

from agent_loop import react_agent

DATA_FILE = "test_data.jsonl"
DEFAULT_OUTPUT = "test_results.jsonl"


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison: lowercase, strip, integers."""
    answer = answer.strip().lower()
    try:
        num = float(answer)
        if num == int(num):
            answer = str(int(num))
    except (ValueError, OverflowError):
        pass
    return answer


async def evaluate_single(question_id: int, question: str) -> dict:
    """Evaluate a single question."""
    start = time.time()
    try:
        answer = await react_agent(question)
        elapsed = time.time() - start
        return {"id": question_id, "answer": answer, "elapsed": round(elapsed, 1), "error": None}
    except Exception as e:
        elapsed = time.time() - start
        return {"id": question_id, "answer": "", "elapsed": round(elapsed, 1), "error": str(e)}


async def main():
    start_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end_id = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    output_file = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUTPUT

    # Load test data (questions + answers in one file)
    data = {}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data[item["id"]] = item

    # Load existing results to support resume
    done_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

    ids_to_process = [i for i in range(start_id, end_id) if i in data]

    print(f"Processing {len(ids_to_process)} questions (IDs {start_id}-{end_id-1})")
    print(f"Skipping {len(done_ids & set(ids_to_process))} already completed")

    correct = 0
    total = 0

    for qid in ids_to_process:
        if qid in done_ids:
            continue

        question = data[qid]["question"]
        expected = data[qid]["answer"]

        print(f"\n{'='*60}")
        print(f"[Q{qid}] {question[:100]}...")

        result = await evaluate_single(qid, question)

        # Append result
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Check accuracy
        predicted_norm = normalize_answer(result["answer"])
        expected_norm = normalize_answer(expected)
        is_correct = predicted_norm == expected_norm
        total += 1
        if is_correct:
            correct += 1

        status = "CORRECT" if is_correct else "WRONG"
        print(f"[Q{qid}] Predicted: {result['answer']}")
        print(f"[Q{qid}] Expected:  {expected}")
        print(f"[Q{qid}] {status} ({result['elapsed']}s)")

        if result["error"]:
            print(f"[Q{qid}] ERROR: {result['error']}")

        if total > 0:
            print(f"Running accuracy: {correct}/{total} = {correct/total:.2%}")

    # Final summary for this run
    print(f"\n{'='*60}")
    if total > 0:
        print(f"FINAL: {correct}/{total} = {correct/total:.2%}")
    else:
        print("No new questions processed this run.")

    # Overall accuracy across all results in output file
    if os.path.exists(output_file):
        all_correct = 0
        all_total = 0
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    qid = item["id"]
                    if qid in data:
                        if normalize_answer(item["answer"]) == normalize_answer(data[qid]["answer"]):
                            all_correct += 1
                        all_total += 1
                except Exception:
                    pass
        if all_total > 0:
            print(f"OVERALL (all results in {output_file}): {all_correct}/{all_total} = {all_correct/all_total:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
