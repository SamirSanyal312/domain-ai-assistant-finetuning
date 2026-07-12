from __future__ import annotations

import argparse
import os
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

# Reduce Hugging Face / tokenizer console noise before importing ML libraries.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@contextmanager
def suppress_library_output():
    """Hide verbose startup banners and non-fatal library messages."""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


# Unsloth must be imported before torch/transformers/peft.
with suppress_library_output():
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    import torch
    from peft import PeftModel
    from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()

from common import generate_answer  # noqa: E402

BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = ROOT / "outputs" / "dpo_adapter"


def load_assistant(adapter_path: Path):
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter not found at {adapter_path}. Run all three training notebooks first "
            "or pass --adapter with the saved DPO adapter path."
        )

    print("Loading NovaDesk model...", flush=True)

    with suppress_library_output():
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )

        # Keep Qwen2.5's text-generation tokens consistent with training.
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.pad_token = "<|endoftext|>"
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        if hasattr(model, "generation_config"):
            model.generation_config.eos_token_id = tokenizer.eos_token_id
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
        )
        FastLanguageModel.for_inference(model)

    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask the NovaDesk DPO-aligned IT helpdesk assistant a question."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask. If omitted, interactive mode starts.",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=DEFAULT_ADAPTER,
        help="Path to the saved DPO LoRA adapter.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Warning: CUDA GPU not detected; 4-bit inference may be unavailable or slow.")

    model, tokenizer = load_assistant(args.adapter)

    if args.question:
        print("\nAssistant:")
        print(generate_answer(model, tokenizer, args.question))
        return

    print("Ready. Type 'exit' to stop.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            print("\nAssistant:")
            print(generate_answer(model, tokenizer, question))


if __name__ == "__main__":
    main()
