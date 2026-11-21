import argparse
import json
import random
import re
import time

from src.config import BASE_DIR
from src.extraction.loader import load_document
from src.processing.chunker import chunk_document

from evaluation.run_eval import normalize

SYSTEM = "You write test questions for a document search system. Reply with JSON only."

PROMPT = """Passage:
{passage}

Write ONE question that this passage answers.
Rules:
1. The question must make sense on its own, without seeing the passage.
2. Paraphrase. Do not reuse the passage's distinctive words in the question.
3. Pick ONE short exact phrase (3 to 6 words) copied letter for letter from the passage that contains the answer.
4. Pick ONE word that a correct answer should contain.

Reply as JSON: {{"question": "...", "evidence": "...", "answer_keyword": "..."}}"""


def parse_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def pick_chunks(chunks, count, seed):
    rng = random.Random(seed)
    usable = [c for c in chunks if len(c["text"].split()) >= 40]
    rng.shuffle(usable)

    picked, taken, pages_used = [], set(), set()
    for index, chunk in enumerate(usable):
        page = chunk["metadata"]["page_number"]
        if page not in pages_used:
            picked.append(chunk)
            taken.add(index)
            pages_used.add(page)
        if len(picked) == count:
            return picked
    for index, chunk in enumerate(usable):
        if index not in taken:
            picked.append(chunk)
            if len(picked) == count:
                break
    return picked


def check_draft(data, chunk_text):
    if not isinstance(data, dict):
        return "reply was not valid JSON"
    question = str(data.get("question", "")).strip()
    evidence = str(data.get("evidence", "")).strip()
    if not 10 <= len(question) <= 250:
        return "question has a strange length"
    words = len(evidence.split())
    if not 2 <= words <= 10:
        return "evidence phrase has a strange length"
    if normalize(evidence) not in normalize(chunk_text):
        return "evidence is not copied exactly from the passage"
    if normalize(evidence) in normalize(question):
        return "question repeats the evidence phrase (too easy)"
    return None


def main(argv=None, llm=None):
    parser = argparse.ArgumentParser(description="Draft test questions from your documents.")
    parser.add_argument("--doc", required=True, help="e.g. data/raw/NOTES2.pdf")
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pause", type=float, default=1.5)
    parser.add_argument("--existing", default="evaluation/test_set.json")
    parser.add_argument("--out", default="evaluation/draft_test_set.json")
    args = parser.parse_args(argv)

    if llm is None:
        from src.llm.groq_client import GroqClient

        llm = GroqClient()

    doc = load_document(str(BASE_DIR / args.doc))
    chunks = chunk_document(doc, lambda t: len(t.split()), max_tokens=120, overlap_tokens=0)
    chosen = pick_chunks(chunks, args.count, args.seed)

    questions, rejected = [], []
    for chunk in chosen:
        reply = llm.generate(PROMPT.format(passage=chunk["text"]), system_instruction=SYSTEM)
        data = parse_json(reply)
        problem = check_draft(data, chunk["text"])
        if problem:
            rejected.append(problem)
        else:
            questions.append(
                {
                    "question": data["question"].strip(),
                    "evidence": [data["evidence"].strip()],
                    "answer_keywords": [str(data.get("answer_keyword", "")).strip()]
                    if data.get("answer_keyword")
                    else [],
                    "answerable": True,
                    "source_page": chunk["metadata"]["page_number"],
                }
            )
        time.sleep(args.pause)

    existing_path = BASE_DIR / args.existing
    unanswerable = []
    if existing_path.exists():
        old = json.loads(existing_path.read_text(encoding="utf-8"))
        unanswerable = [q for q in old.get("questions", []) if not q.get("answerable", True)]

    out = {"documents": [args.doc], "questions": questions + unanswerable}
    out_path = BASE_DIR / args.out
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Drafted {len(questions)} questions, rejected {len(rejected)}.")
    for reason in sorted(set(rejected)):
        print(f"  rejected: {reason} (x{rejected.count(reason)})")
    print(f"Kept {len(unanswerable)} unanswerable question(s) from {args.existing}.")
    print(f"Saved to {args.out}. READ EVERY QUESTION before using it.")


if __name__ == "__main__":
    main()