from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from unsloth import FastLanguageModel

from common import generate_answer

BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
DEFAULT_ADAPTER = Path("outputs/dpo_adapter")


def load_assistant(adapter_path: Path):
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter not found at {adapter_path}. Run the three training notebooks first "
            "or pass --adapter with the downloaded DPO adapter path."
        )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the NovaDesk DPO-aligned IT helpdesk assistant a question.")
    parser.add_argument("question", nargs="?", help="Question to ask. If omitted, interactive mode starts.")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER, help="Path to the saved DPO LoRA adapter.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Warning: CUDA GPU not detected. 4-bit inference may be slow or unavailable in this environment.")
    model, tokenizer = load_assistant(args.adapter)

    if args.question:
        print(generate_answer(model, tokenizer, args.question))
        return

    print("NovaDesk assistant. Type 'exit' to stop.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            print("Assistant:", generate_answer(model, tokenizer, question))


if __name__ == "__main__":
    main()
