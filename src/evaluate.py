"""Chunk-size experiment.

Step 1 - run every question in questions.json at 200, 500 and 1000 words:
    python src/evaluate.py run
  -> writes results/chunk_size_eval.csv with an empty `correct` column.

Step 2 - open the CSV, put y or n in `correct` for every row (score by hand).

Step 3 - summarize and write the table into README.md:
    python src/evaluate.py summarize
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from langchain_rag import build_chains, flush_langfuse, langfuse_handler, run_config, sources_of

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = ROOT / "questions.json"
RESULTS_FILE = ROOT / "results" / "chunk_size_eval.csv"
README = ROOT / "README.md"
CHUNK_SIZES = [200, 500, 1000]
FIELDS = ["chunk_size", "id", "question", "expected_answer", "answer", "sources", "correct"]


def load_questions():
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    missing = [q["id"] for q in questions if not q["question"].strip() or not q["expected_answer"].strip()]
    if missing:
        raise SystemExit(f"Fill in question and expected_answer in questions.json for ids: {missing}")
    return questions


def run(force=False):
    if RESULTS_FILE.exists() and not force:
        raise SystemExit(f"{RESULTS_FILE} already exists (it may hold your scores). Use --force to overwrite.")

    questions = load_questions()
    handler = langfuse_handler()
    rows = []

    try:
        for chunk_size in CHUNK_SIZES:
            overlap = max(10, chunk_size // 10)
            rag_chain, _ = build_chains(chunk_size, overlap)
            for q in questions:
                print(f"[{chunk_size}w] Q{q['id']}: {q['question']}")
                result = rag_chain.invoke(
                    q["question"],
                    config=run_config(handler, "eval", tags=[f"chunk-{chunk_size}w", "eval"]),
                )
                rows.append({
                    "chunk_size": chunk_size,
                    "id": q["id"],
                    "question": q["question"],
                    "expected_answer": q["expected_answer"],
                    "answer": result["answer"],
                    "sources": " | ".join(sources_of(result["docs"])),
                    "correct": "",
                })
    finally:
        flush_langfuse()

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {RESULTS_FILE}. Score the `correct` column with y/n, then run `summarize`.")


def summarize():
    with open(RESULTS_FILE, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    unscored = [r for r in rows if r["correct"].strip().lower() not in {"y", "n", "yes", "no", "1", "0"}]
    if unscored:
        raise SystemExit(f"{len(unscored)} rows are not scored yet (put y or n in the `correct` column).")

    correct = defaultdict(int)
    total = defaultdict(int)
    for row in rows:
        size = int(row["chunk_size"])
        total[size] += 1
        correct[size] += row["correct"].strip().lower() in {"y", "yes", "1"}

    lines = [
        "| Chunk size (words) | Overlap (words) | Correct | Accuracy |",
        "|---|---|---|---|",
    ]
    for size in sorted(total):
        lines.append(
            f"| {size} | {max(10, size // 10)} | {correct[size]} / {total[size]} | "
            f"{correct[size] / total[size]:.0%} |"
        )
    table = "\n".join(lines)
    print(table)

    readme = README.read_text(encoding="utf-8")
    pattern = re.compile(r"(<!-- chunk-table:start -->\n).*?(\n<!-- chunk-table:end -->)", re.S)
    README.write_text(pattern.sub(lambda m: m.group(1) + table + m.group(2), readme), encoding="utf-8")
    print("\nREADME.md table updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk-size comparison")
    parser.add_argument("command", choices=["run", "summarize"])
    parser.add_argument("--force", action="store_true", help="overwrite an existing results file")
    args = parser.parse_args()

    if args.command == "run":
        run(args.force)
    else:
        summarize()
