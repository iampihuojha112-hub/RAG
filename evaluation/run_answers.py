import argparse
import json
import time

from src.config import BASE_DIR
from src.llm.prompt_builder import NOT_FOUND_MESSAGE


def main(argv=None, pipeline=None):
    parser = argparse.ArgumentParser(description="Check real answers end to end.")
    parser.add_argument("--test-set", default="evaluation/test_set.json")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between questions")
    args = parser.parse_args(argv)

    if pipeline is None:
        from src.pipeline.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline()

    data = json.loads((BASE_DIR / args.test_set).read_text(encoding="utf-8"))
    for relative in data["documents"]:
        pipeline.ingest(str(BASE_DIR / relative))

    good_answers = good_refusals = answerable = unanswerable = 0

    for number, q in enumerate(data["questions"], start=1):
        result = pipeline.ask(q["question"])
        answer = result["answer"]
        refused = (not result["found"]) or NOT_FOUND_MESSAGE.lower() in answer.lower()
        cited = any(s["cited"] for s in result["sources"])
        problems = []

        if q.get("answerable", True):
            answerable += 1
            keywords = [k.lower() for k in q.get("answer_keywords", [])]
            if refused:
                problems.append("refused, but it should have answered")
            elif not cited:
                problems.append("answer has no citation")
            elif keywords and not any(k in answer.lower() for k in keywords):
                problems.append("none of the expected keywords appear")
            good_answers += not problems
        else:
            unanswerable += 1
            if not refused:
                problems.append("answered, but it should have refused")
            good_refusals += not problems

        status = "PASS" if not problems else "FAIL"
        print(f"[{number:>2}] {status}  {q['question']}")
        if problems:
            print(f"       problem: {'; '.join(problems)}")
            print(f"       answer : {answer[:200]}")
        time.sleep(args.pause)

    print("\n=== SUMMARY ===")
    print(f"answerable questions answered correctly : {good_answers}/{answerable}")
    print(f"unanswerable questions refused          : {good_refusals}/{unanswerable}")


if __name__ == "__main__":
    main()