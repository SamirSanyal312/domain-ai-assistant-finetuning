from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

SYSTEM_PROMPT = """You are NovaDesk, an internal IT helpdesk assistant. Give safe, concise, policy-grounded troubleshooting steps. Never request or expose passwords, one-time codes, private keys, full payment-card numbers, health records, or unredacted identity documents. State when the user should stop troubleshooting and call the Service Desk or Security Operations. Do not invent outage restoration times."""


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return rows


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 220) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def write_comparison_report(path: str | Path, title: str, questions: Iterable[str], columns: list[tuple[str, list[str]]]) -> None:
    questions = list(questions)
    header = ["Question"] + [name for name, _ in columns]
    lines = [f"# {title}", "", "| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for idx, question in enumerate(questions):
        row = [markdown_escape(question)]
        for _, values in columns:
            row.append(markdown_escape(values[idx] if idx < len(values) else "Not generated"))
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(["", "## Evaluation note", "", "Review correctness, domain accuracy, clarity, safety, helpfulness, tone, and hallucination risk. Replace any automated judgment with a short human rationale before submission."])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
