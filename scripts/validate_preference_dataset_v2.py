from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "preference_dataset_v2.jsonl"
HELD_OUT = ROOT / "data" / "evaluation_questions.json"

def normalize(text: str) -> str:
    return " ".join(text.lower().split())

def main() -> None:
    rows = [
        json.loads(line)
        for line in DATA.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    required = {"category", "split", "prompt", "chosen", "rejected"}
    assert len(rows) >= 50, f"Need at least 50 pairs, found {len(rows)}"
    assert all(required <= set(row) for row in rows), "Missing required fields"
    assert all(row["split"] in {"train", "eval"} for row in rows)
    assert all(normalize(row["chosen"]) != normalize(row["rejected"]) for row in rows)

    prompts = [normalize(row["prompt"]) for row in rows]
    assert len(prompts) == len(set(prompts)), "Duplicate prompts found"

    triples = [
        (normalize(row["prompt"]), normalize(row["chosen"]), normalize(row["rejected"]))
        for row in rows
    ]
    assert len(triples) == len(set(triples)), "Duplicate preference triples found"

    train_categories = {row["category"] for row in rows if row["split"] == "train"}
    eval_categories = {row["category"] for row in rows if row["split"] == "eval"}
    assert train_categories == eval_categories, "Each category must appear in train and eval"

    if HELD_OUT.exists():
        questions = json.loads(HELD_OUT.read_text(encoding="utf-8"))
        overlap = {normalize(q) for q in questions}.intersection(prompts)
        assert not overlap, f"Exact held-out prompt leakage: {sorted(overlap)}"

    chosen_counts = Counter(normalize(row["chosen"]) for row in rows)
    rejected_counts = Counter(normalize(row["rejected"]) for row in rows)

    print("Validation passed")
    print("Rows:", len(rows))
    print("Train:", sum(row["split"] == "train" for row in rows))
    print("Eval:", sum(row["split"] == "eval" for row in rows))
    print("Categories:", len(train_categories))
    print("Unique prompts:", len(set(prompts)))
    print("Unique chosen:", len(chosen_counts))
    print("Unique rejected:", len(rejected_counts))
    print("Largest exact chosen duplication:", max(chosen_counts.values()))
    print("Largest exact rejected duplication:", max(rejected_counts.values()))

if __name__ == "__main__":
    main()
