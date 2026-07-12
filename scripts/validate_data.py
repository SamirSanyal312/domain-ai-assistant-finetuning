from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    rows=[]
    for i,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if line.strip():
            rows.append(json.loads(line))
    return rows

raw=[p for p in (ROOT/"data/non_instruction_data.txt").read_text(encoding="utf-8").split("\n\n") if p.strip()]
sft=load(ROOT/"data/instruction_dataset.jsonl")
pref=load(ROOT/"data/preference_dataset.jsonl")
assert len(raw)>=50, len(raw)
assert len(sft)>=100, len(sft)
assert len(pref)>=50, len(pref)
assert all(r.get("instruction") and r.get("response") for r in sft)
assert all(r.get("prompt") and r.get("chosen") and r.get("rejected") for r in pref)
assert len({r["instruction"] for r in sft})==len(sft), "Duplicate instructions"
assert all(r["chosen"] != r["rejected"] for r in pref)
for secret in ["send your password", "share the password"]:
    assert all(secret not in r["response"].lower() for r in sft)
print({"raw_paragraphs":len(raw),"instruction_examples":len(sft),"preference_examples":len(pref)})
print("Top SFT topics:", Counter(r.get("topic") for r in sft).most_common(5))
print("Dataset validation passed.")
