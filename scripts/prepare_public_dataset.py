from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "Tobi-Bueck/customer-support-tickets"
ALLOWED_QUEUES = {"IT Support", "Technical Support", "Service Outages and Maintenance"}


def clean(text: str) -> str:
    text = re.sub(r"<name>", "user", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/public_ticket_sample.jsonl"))
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    dataset = load_dataset(DATASET_ID, split="train")
    dataset = dataset.filter(
        lambda row: row.get("language") == "en"
        and row.get("queue") in ALLOWED_QUEUES
        and bool(row.get("body"))
        and bool(row.get("answer"))
    )
    dataset = dataset.shuffle(seed=42).select(range(min(args.limit, len(dataset))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in dataset:
            result = {
                "subject": clean(row.get("subject", "")),
                "body": clean(row["body"]),
                "answer": clean(row["answer"]),
                "queue": row.get("queue"),
                "priority": row.get("priority"),
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Wrote {len(dataset)} cleaned rows to {args.output}")
    print("License note: the source dataset card lists CC BY-NC 4.0; retain attribution and review the license before reuse.")


if __name__ == "__main__":
    main()
