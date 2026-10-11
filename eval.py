"""
Batch evaluation script for Research Agent.
Runs questions from question.jsonl and compares with reference answers.
"""
import asyncio
import json
import os
import sys
import time
import re

# Load env before importing agent
from dotenv import load_dotenv
load_dotenv()

from agent_loop import react_agent


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison: lowercase, strip, integers."""
    answer = answer.strip().lower()
    # Try to convert to integer if it looks like a number
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
        return {
            "id": question_id,
            "answer": answer,
            "elapsed": round(elapsed, 1),
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "id": question_id,
            "answer": "",
            "elapsed": round(elapsed, 1),
            "error": str(e),
        }


async def main():
    # Parse arguments
    start_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end_id = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    output_file = sys.argv[3] if len(sys.argv) > 3 else "my_results.jsonl"

    # Load questions
    questions = {}
    with open("question.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            questions[item["id"]] = item["question"]

    # Load reference answers (if available)
    ref_answers = {}
    if os.path.exists("submit_results.jsonl"):
        with open("submit_results.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                ref_answers[item["id"]] = item["answer"]

    # Load existing results to skip already done
    done_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    done_ids.add(item["id"])
                except:
                    pass

    # Run evaluation
    correct = 0
    total = 0
    skipped = 0

    ids_to_process = [i for i in range(start_id, end_id) if i in questions and i != 16]  # Skip Q16 (voided)

    print(f"Processing {len(ids_to_process)} questions (IDs {start_id}-{end_id-1})")
    print(f"Skipping {len(done_ids)} already completed")

    for qid in ids_to_process:
        if qid in done_ids:
            skipped += 1
            # Still count for accuracy if we have the result
            continue

        question = questions[qid]
        print(f"\n{'='*60}")
        print(f"[Q{qid}] {question[:100]}...")

        result = await evaluate_single(qid, question)

        # Write result
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Check accuracy
        if qid in ref_answers:
            predicted = normalize_answer(result["answer"])
            expected = normalize_answer(ref_answers[qid])
            is_correct = predicted == expected
            total += 1
            if is_correct:
                correct += 1
            status = "CORRECT" if is_correct else "WRONG"
            print(f"[Q{qid}] Predicted: {result['answer']}")
            print(f"[Q{qid}] Expected:  {ref_answers[qid]}")
            print(f"[Q{qid}] {status} ({result['elapsed']}s)")
        else:
            total += 1
            print(f"[Q{qid}] Answer: {result['answer']} ({result['elapsed']}s)")

        if result["error"]:
            print(f"[Q{qid}] ERROR: {result['error']}")

        print(f"Running accuracy: {correct}/{total} = {correct/total:.2%}" if total > 0 else "")

    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL: {correct}/{total} = {correct/total:.2%}" if total > 0 else "No results")

    # Also compute accuracy including previously done results
    if done_ids and os.path.exists(output_file):
        all_correct = 0
        all_total = 0
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    qid = item["id"]
                    if qid in ref_answers and qid != 16:
                        predicted = normalize_answer(item["answer"])
                        expected = normalize_answer(ref_answers[qid])
                        all_total += 1
                        if predicted == expected:
                            all_correct += 1
                except:
                    pass
        if all_total > 0:
            print(f"OVERALL (all results): {all_correct}/{all_total} = {all_correct/all_total:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
