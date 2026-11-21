import argparse
import json
import statistics
import tempfile
from pathlib import Path

from src.config import (
    BASE_DIR,
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    EMBEDDING_MODEL_NAME,
    MAX_DISTANCE,
    TOP_K,
)
from src.extraction.loader import load_document
from src.processing.chunker import chunk_document
from src.retrieval.factory import make_retriever
from src.vectorstore.chroma_store import ChromaStore

DEFAULT_CONFIGS = "100:20,150:30,200:40"
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
RANK_DEPTH = 10


def normalize(text):
    return " ".join(text.lower().split())


def pct(count, total):
    return f"{100 * count / total:5.0f}%" if total else "  n/a"


def parse_configs(text):
    configs = []
    for part in text.split(","):
        size, overlap = part.split(":")
        configs.append((int(size), int(overlap)))
    current = (CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS)
    if current not in configs:
        configs.append(current)
    return configs


def load_test_set(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    docs = [load_document(str(BASE_DIR / p)) for p in data["documents"]]
    return docs, data["questions"]


def check_questions(questions, docs):
    page_texts = [normalize(p["text"]) for d in docs for p in d["pages"]]
    usable = []
    for q in questions:
        if not q.get("answerable", True):
            usable.append(q)
            continue
        evidence = [normalize(e) for e in q.get("evidence", [])]
        if not evidence:
            print(f"SKIPPED (no evidence given): {q['question']}")
        elif not any(e in t for e in evidence for t in page_texts):
            print(f"SKIPPED (evidence not found in the document text): {q['question']}")
        else:
            usable.append({**q, "evidence": evidence})
    return usable


def build_store(docs, embedder, max_tokens, overlap, folder):
    store = ChromaStore(path=str(Path(folder) / f"db_{max_tokens}_{overlap}"))
    for doc in docs:
        chunks = chunk_document(doc, embedder.count_tokens, max_tokens, overlap)
        if chunks:
            store.add_chunks(chunks, embedder.embed_texts([c["text"] for c in chunks]))
    return store


def first_hit(hits, evidence):
    for rank, hit in enumerate(hits, start=1):
        text = normalize(hit["text"])
        if any(e in text for e in evidence):
            return rank, hit["distance"]
    return None, None


def spread(values):
    if not values:
        return "n/a"
    return f"min {min(values):.2f}  median {statistics.median(values):.2f}  max {max(values):.2f}"


def evaluate(retriever, store, questions, show_misses):
    answerable = [q for q in questions if q.get("answerable", True)]
    unanswerable = [q for q in questions if not q.get("answerable", True)]

    results = []
    for q in answerable:
        hits = retriever.search(q["question"], top_k=RANK_DEPTH)
        rank, distance = first_hit(hits, q["evidence"])
        results.append((q, hits, rank, distance))

    n = len(answerable)
    print(f"  chunks stored: {store.count()}")
    if n:
        ranks = [r for _, _, r, _ in results]
        hit = lambda k: sum(1 for r in ranks if r is not None and r <= k)
        mrr = sum(1 / r for r in ranks if r) / n
        print(f"  answerable questions: {n}")
        print(f"  hit@1 {pct(hit(1), n)}   hit@3 {pct(hit(3), n)}   hit@5 {pct(hit(5), n)}   MRR {mrr:.2f}")

    closest_unanswerable = []
    for q in unanswerable:
        hits = retriever.search(q["question"], top_k=TOP_K)
        closest_unanswerable.append(min((h["distance"] for h in hits), default=9.9))

    print(f"  unanswerable questions: {len(unanswerable)}")
    print(f"  distance of the right chunk (answerable) : {spread([d for _, _, r, d in results if r])}")
    print(f"  closest distance (unanswerable)          : {spread(closest_unanswerable)}")
    print(f"  {'max_distance':>12} | {'answerable kept':>15} | {'unanswerable refused':>20} | balance")
    for thr in sorted(set(THRESHOLDS + [MAX_DISTANCE])):
        kept = sum(1 for _, _, r, d in results if r is not None and r <= TOP_K and d <= thr)
        refused = sum(1 for d in closest_unanswerable if d > thr)
        parts = []
        if n:
            parts.append(kept / n)
        if unanswerable:
            parts.append(refused / len(unanswerable))
        balance = f"{100 * sum(parts) / len(parts):4.0f}%" if parts else " n/a"
        mark = "   <-- current" if thr == MAX_DISTANCE else ""
        print(f"  {thr:>12} | {pct(kept, n):>15} | {pct(refused, len(unanswerable)):>20} | {balance}{mark}")

    misses = [(q, hits) for q, hits, r, _ in results if r is None or r > TOP_K]
    if misses:
        print(f"  MISSES (no evidence in top {TOP_K}): {len(misses)}")
        for q, hits in misses[:show_misses]:
            top = hits[0] if hits else None
            print(f"   - {q['question']}")
            print(f"       looked for: {q['evidence'][0]}")
            if top:
                print(f"       top chunk (distance {top['distance']:.2f}): {top['text'][:140]}")

    return {
        "hit5": sum(1 for _, _, r, _ in results if r is not None and r <= 5) / n if n else None,
        "mrr": sum(1 / r for _, _, r, _ in results if r) / n if n else None,
    }


def warn_if_chunks_too_long(embedder, max_tokens):
    limit = getattr(embedder, "max_tokens", None)
    if limit is None:
        return
    prefix = getattr(embedder, "document_prefix", "")
    extra = embedder.count_tokens(prefix) if prefix else 0
    if max_tokens + extra + 2 > limit:
        print(
            f"  WARNING: chunks of {max_tokens} tokens do not fit in this model's "
            f"limit ({limit}). The end of each chunk will be ignored."
        )


def main(argv=None, embedder=None, reranker=None):
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--test-set", default="evaluation/test_set.json")
    parser.add_argument("--configs", default=DEFAULT_CONFIGS, help="max_tokens:overlap pairs")
    parser.add_argument("--modes", default="vector", help="vector, hybrid, vector+rerank, hybrid+rerank")
    parser.add_argument("--embedders", default=EMBEDDING_MODEL_NAME, help="comma-separated model names")
    parser.add_argument("--details", action="store_true", help="show every miss")
    args = parser.parse_args(argv)
    modes = [m.strip() for m in args.modes.split(",")]
    model_names = [m.strip() for m in args.embedders.split(",")]

    if reranker is None and any(m.endswith("+rerank") for m in modes):
        from src.retrieval.reranker import Reranker

        reranker = Reranker()

    docs, questions = load_test_set(BASE_DIR / args.test_set)
    questions = check_questions(questions, docs)
    if not questions:
        raise SystemExit("No usable questions. Fix evaluation/test_set.json.")

    show_misses = 999 if args.details else 5
    summary = []

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        for model_name in model_names:
            if embedder is not None:
                current_embedder = embedder
            else:
                from src.embeddings.embedder import Embedder

                current_embedder = Embedder(model_name)

            for max_tokens, overlap in parse_configs(args.configs):
                store = build_store(docs, current_embedder, max_tokens, overlap, Path(folder) / model_name.replace("/", "_"))
                for mode in modes:
                    current = "  (current chunk settings)" if (max_tokens, overlap) == (CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS) else ""
                    print(f"\n=== model {model_name} | chunk size {max_tokens}, overlap {overlap} | mode {mode}{current} ===")
                    warn_if_chunks_too_long(current_embedder, max_tokens)
                    retriever = make_retriever(mode, current_embedder, store, reranker)
                    scores = evaluate(retriever, store, questions, show_misses)
                    summary.append((model_name, max_tokens, overlap, mode, store.count(), scores))

    print("\n=== SUMMARY ===")
    print(f"{'model':<24} {'size':>5} {'overlap':>8} {'mode':>13} {'chunks':>7} {'hit@5':>7} {'MRR':>6}")
    for model_name, max_tokens, overlap, mode, count, scores in summary:
        hit5 = pct(round(scores["hit5"] * 100), 100) if scores["hit5"] is not None else "  n/a"
        mrr = f"{scores['mrr']:.2f}" if scores["mrr"] is not None else " n/a"
        print(f"{model_name:<24} {max_tokens:>5} {overlap:>8} {mode:>13} {count:>7} {hit5:>7} {mrr:>6}")


if __name__ == "__main__":
    main()