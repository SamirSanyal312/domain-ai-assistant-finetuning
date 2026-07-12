from __future__ import annotations

import argparse
import os
import threading
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Unsloth must be imported before Transformers / PEFT.
from unsloth import FastLanguageModel

import gradio as gr
import torch
from peft import PeftModel
from transformers import TextIteratorStreamer
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = os.getenv(
    "NOVADESK_BASE_MODEL",
    "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
)

SYSTEM_PROMPT = """You are NovaDesk, an educational internal IT helpdesk assistant.
Give concise, safe, professional first-line support. Never request or accept passwords,
one-time codes, private keys, recovery codes, or other secrets. Do not claim access to
accounts, devices, tickets, or company systems. Escalate security incidents, stolen devices,
major outages, and access-control decisions through approved support channels. Answer the
user's exact question rather than returning a loosely related troubleshooting template."""

MODES = {
    "Base Model": "base",
    "SFT Model": "sft_mode",
    "DPO Model": "dpo_mode",
}

DESCRIPTIONS = {
    "Base Model": "Original Qwen2.5 model with NovaDesk adapters disabled.",
    "SFT Model": "Supervised fine-tuned NovaDesk adapter.",
    "DPO Model": "Preference-aligned NovaDesk adapter.",
}

CSS = """
:root,
.gradio-container {
    --body-background-fill: #f4f7fb;
    --body-text-color: #172033;
    --body-text-color-subdued: #65738a;

    --block-background-fill: #ffffff;
    --block-border-color: #dbe4f0;
    --block-label-background-fill: #eef4ff;
    --block-label-text-color: #2457d6;
    --block-title-text-color: #172033;

    --input-background-fill: #ffffff;
    --input-border-color: #cbd7e6;
    --input-border-color-focus: #4f7cff;
    --input-placeholder-color: #91a0b5;

    --button-primary-background-fill: #356df3;
    --button-primary-background-fill-hover: #2457d6;
    --button-primary-text-color: #ffffff;

    --button-secondary-background-fill: #ffffff;
    --button-secondary-background-fill-hover: #f3f6fb;
    --button-secondary-border-color: #cbd7e6;
    --button-secondary-text-color: #172033;

    --checkbox-background-color-selected: #356df3;
    --checkbox-border-color-selected: #356df3;
}

html,
body,
.gradio-container {
    background: #f4f7fb !important;
    color: #172033 !important;
}

.gradio-container {
    max-width: 1480px !important;
    margin: 0 auto !important;
    padding: 24px 28px 36px !important;
}

/* Header */
#hero {
    padding: 25px 29px;
    margin-bottom: 18px;
    border: 1px solid rgba(85, 132, 218, 0.34);
    border-radius: 22px;
    background:
        radial-gradient(circle at 88% 15%, rgba(104, 126, 255, 0.28), transparent 26%),
        linear-gradient(135deg, #10284a 0%, #173b68 58%, #253b78 100%);
    box-shadow: 0 16px 42px rgba(26, 52, 92, 0.18);
}

#hero h1 {
    margin: 0 0 7px;
    color: #ffffff;
    font-size: 32px;
    font-weight: 760;
    letter-spacing: -0.02em;
}

#hero p {
    margin: 0;
    color: #d8e5ff;
    font-size: 15px;
}

/* Main panels */
.nd-card {
    border: 1px solid #dbe4f0 !important;
    border-radius: 20px !important;
    background: #ffffff !important;
    box-shadow: 0 12px 34px rgba(42, 64, 99, 0.10);
}

.nd-card,
.nd-card p,
.nd-card span,
.nd-card label,
.nd-card h1,
.nd-card h2,
.nd-card h3,
.nd-card h4,
.nd-card .prose {
    color: #172033;
}

.nd-card .info,
.nd-card .secondary-wrap,
.nd-card small {
    color: #65738a !important;
}

/* Left controls */
#controls-panel {
    padding: 18px !important;
}

#controls-panel h2 {
    color: #172033 !important;
    margin-bottom: 12px;
}

#controls-panel fieldset {
    background: #f8faff !important;
    border: 1px solid #dce6f4 !important;
    border-radius: 15px !important;
}

#mode-status {
    padding: 14px 15px;
    margin-top: 12px;
    border: 1px solid #d8e4f5;
    border-left: 4px solid #356df3;
    border-radius: 14px;
    background: linear-gradient(135deg, #f6f9ff, #eef4ff) !important;
    color: #172033 !important;
}

#mode-status * {
    color: #172033 !important;
}

#mode-status code {
    display: block;
    margin-top: 5px;
    padding: 7px 9px;
    overflow-wrap: anywhere;
    border: 1px solid #d7e1ef;
    border-radius: 8px;
    background: #ffffff;
    color: #33445f !important;
    font-size: 11px;
}

/* Accordions and controls */
.gradio-container .accordion {
    border: 1px solid #dbe4f0 !important;
    border-radius: 13px !important;
    background: #ffffff !important;
}

.gradio-container input,
.gradio-container textarea {
    color: #172033 !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #91a0b5 !important;
}

/* Chat surface */
#conversation-panel {
    padding: 0 !important;
    overflow: hidden;
}

#chatbot {
    min-height: 610px;
    border: 1px solid #dbe4f0 !important;
    border-radius: 18px !important;
    background:
        linear-gradient(rgba(248, 250, 253, 0.96), rgba(248, 250, 253, 0.96)),
        radial-gradient(circle at 50% 0%, #eaf1ff, transparent 48%) !important;
}

#chatbot .message {
    border-radius: 14px !important;
    box-shadow: 0 4px 14px rgba(43, 63, 96, 0.07);
}

#chatbot .message.user {
    background: #356df3 !important;
    color: #ffffff !important;
}

#chatbot .message.bot,
#chatbot .message.assistant {
    border: 1px solid #dbe4f0 !important;
    background: #ffffff !important;
    color: #172033 !important;
}

/* Composer */
#message-box {
    border: 1px solid #dbe4f0 !important;
    border-radius: 15px !important;
    background: #ffffff !important;
    box-shadow: 0 6px 18px rgba(45, 66, 101, 0.08);
}

#send {
    min-height: 48px;
    border: 0 !important;
    border-radius: 13px !important;
    background: linear-gradient(135deg, #356df3, #6c56ef) !important;
    color: #ffffff !important;
    font-weight: 720;
    box-shadow: 0 7px 18px rgba(72, 89, 220, 0.25);
    transition: transform 0.16s ease, box-shadow 0.16s ease;
}

#send:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 24px rgba(72, 89, 220, 0.32);
}

/* Example prompt buttons */
.gradio-container button {
    border-radius: 11px;
}

.gradio-container button:not(.primary) {
    color: #26364f;
}

/* Footer */
footer,
footer a {
    color: #718096 !important;
}

/* Responsive layout */
@media (max-width: 900px) {
    .gradio-container {
        padding: 14px !important;
    }

    #hero {
        padding: 21px;
    }

    #hero h1 {
        font-size: 27px;
    }

    #chatbot {
        min-height: 500px;
    }
}
"""


def resolve_adapter(explicit: str | None, candidates: list[Path], label: str) -> Path:
    paths = ([Path(explicit).expanduser()] if explicit else []) + candidates
    checked: list[Path] = []
    for raw in paths:
        path = raw if raw.is_absolute() else PROJECT_ROOT / raw
        checked.append(path)
        if (path / "adapter_config.json").exists():
            return path
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_dir() and (child / "adapter_config.json").exists():
                    return child
    locations = "\n".join(f"  - {p}" for p in checked)
    raise FileNotFoundError(
        f"{label} adapter not found. Checked:\n{locations}\n"
        "Copy the adapter from Google Drive or pass its path explicitly."
    )


class ModelManager:
    def __init__(self, sft_path: Path, dpo_path: Path, max_seq_length: int) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU not detected. This app loads an Unsloth 4-bit model; "
                "run it in a supported local GPU environment."
            )

        self.sft_path = sft_path
        self.dpo_path = dpo_path
        self.lock = threading.RLock()

        print("Loading base model...")
        base, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.padding_side = "left"
        base.config.eos_token_id = tokenizer.eos_token_id
        base.config.pad_token_id = tokenizer.pad_token_id
        base.config.use_cache = True
        if hasattr(base, "generation_config"):
            base.generation_config.eos_token_id = tokenizer.eos_token_id
            base.generation_config.pad_token_id = tokenizer.pad_token_id
            base.generation_config.max_length = None

        print(f"Loading SFT adapter: {sft_path}")
        model = PeftModel.from_pretrained(
            base, str(sft_path), adapter_name="sft_mode", is_trainable=False
        )
        print(f"Loading DPO adapter: {dpo_path}")
        model.load_adapter(str(dpo_path), adapter_name="dpo_mode", is_trainable=False)
        model.eval()
        FastLanguageModel.for_inference(model)

        self.model = model
        self.tokenizer = tokenizer
        self.device = model.get_input_embeddings().weight.device
        print("Ready. Adapters:", list(model.peft_config.keys()))

    @contextmanager
    def selected_mode(self, label: str):
        name = MODES[label]
        if name == "base":
            with self.model.disable_adapter():
                yield
            return
        try:
            self.model.set_adapter(name, inference_mode=True)
        except TypeError:
            self.model.set_adapter(name)
        self.model.eval()
        yield

    def status(self, label: str) -> str:
        source = (
            "No adapter active"
            if label == "Base Model"
            else f"`{self.sft_path if label == 'SFT Model' else self.dpo_path}`"
        )
        return f"### Active mode: **{label}**\n{DESCRIPTIONS[label]}\n\n**Source:** {source}"

    def stream(
        self,
        message: str,
        history: list[dict],
        mode: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        history_turns: int,
    ) -> Iterator[str]:
        prior = history[-history_turns * 2 :] if history_turns else []
        messages = [{"role": "system", "content": system_prompt.strip()}]
        messages += [
            {"role": x["role"], "content": str(x["content"])}
            for x in prior
            if x.get("role") in {"user", "assistant"} and x.get("content")
        ]
        messages.append({"role": "user", "content": message.strip()})

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=180
        )
        do_sample = temperature > 0
        kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "repetition_penalty": 1.08,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }
        if do_sample:
            kwargs.update(temperature=max(temperature, 0.05), top_p=top_p)

        with self.lock, torch.inference_mode(), self.selected_mode(mode):
            thread = threading.Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
            thread.start()
            answer = ""
            for piece in streamer:
                answer += piece
                yield answer.strip()
            thread.join(timeout=5)


def build_ui(manager: ModelManager) -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo", neutral_hue="slate")
    with gr.Blocks(theme=theme, css=CSS, title="NovaDesk Model Lab", analytics_enabled=False) as demo:
        gr.HTML(
            '<section id="hero"><h1>NovaDesk Model Lab</h1>'
            '<p>Compare Base, SFT, and DPO behavior from one local dashboard.</p></section>'
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=310, elem_id="controls-panel", elem_classes="nd-card"):
                gr.Markdown("## Model controls")
                mode = gr.Radio(list(MODES), value="DPO Model", label="Inference mode",
                                info="Changing modes clears chat for a fair comparison.")
                status = gr.Markdown(manager.status("DPO Model"), elem_id="mode-status")
                with gr.Accordion("Generation settings", open=False):
                    temperature = gr.Slider(0, 1.2, value=0.15, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                    max_tokens = gr.Slider(64, 512, value=220, step=16, label="Maximum new tokens")
                    history_turns = gr.Slider(0, 10, value=4, step=1, label="History turns")
                with gr.Accordion("System prompt", open=False):
                    system = gr.Textbox(value=SYSTEM_PROMPT, lines=9, label="System instruction")
                clear = gr.Button("Clear conversation")
                gr.Markdown("**Fair comparison:** use temperature `0` and ask the same prompt in every mode.")

            with gr.Column(scale=3, elem_id="conversation-panel", elem_classes="nd-card"):
                chatbot = gr.Chatbot(
    value=[
        {
            "role": "assistant",
            "content": (
                "Select a model mode and ask an IT helpdesk question. "
                "The model loads once; only the active adapter changes."
            ),
        }
    ],
    elem_id="chatbot",
    label="NovaDesk conversation",
    height=620,
    buttons=["copy"],
)
                with gr.Row():
                    message = gr.Textbox(
                        placeholder="Ask about VPN, MFA, phishing, devices, access...",
                        show_label=False, lines=2, scale=8, autofocus=True, elem_id="message-box"
                    )
                    send = gr.Button("Send", variant="primary", scale=1, elem_id="send")
                gr.Examples(
                    examples=[
                        ["I entered my corporate password on a phishing page. What should I do?"],
                        ["Half of the company cannot connect to VPN. How should this be prioritized?"],
                        ["I lost my MFA phone and have no backup factor."],
                        ["Can support copy my coworker's shared-drive permissions to me?"],
                        ["My company laptop was stolen. Can I report it tomorrow?"],
                        ["Can I send you my password so you can troubleshoot my account?"],
                    ],
                    inputs=message,
                    label="Evaluation prompts",
                )

        def respond(msg, history, selected_mode, system_prompt, temp, p, tokens, turns):
            if not msg or not msg.strip():
                yield history or [], ""
                return
            prior = list(history or [])
            visible = prior + [
                {"role": "user", "content": msg.strip()},
                {"role": "assistant", "content": ""},
            ]
            for partial in manager.stream(msg, prior, selected_mode, system_prompt, temp, p, tokens, turns):
                visible[-1] = {"role": "assistant", "content": partial}
                yield visible, ""

        def switch_mode(selected_mode):
            return ([{"role": "assistant", "content": f"**{selected_mode} selected.** Chat cleared."}],
                    manager.status(selected_mode))

        def clear_chat(selected_mode):
            return [{"role": "assistant", "content": f"Conversation cleared. Using **{selected_mode}**."}]

        inputs = [message, chatbot, mode, system, temperature, top_p, max_tokens, history_turns]
        message.submit(respond, inputs=inputs, outputs=[chatbot, message], concurrency_limit=1)
        send.click(respond, inputs=inputs, outputs=[chatbot, message], concurrency_limit=1)
        mode.change(switch_mode, inputs=mode, outputs=[chatbot, status])
        clear.click(clear_chat, inputs=mode, outputs=chatbot)
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NovaDesk local Base/SFT/DPO frontend")
    parser.add_argument("--sft-adapter", default=os.getenv("SFT_ADAPTER_PATH"))
    parser.add_argument("--dpo-adapter", default=os.getenv("DPO_ADAPTER_PATH"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sft = resolve_adapter(args.sft_adapter, [PROJECT_ROOT / "outputs/sft_adapter_v2", PROJECT_ROOT / "outputs/sft_adapter"], "SFT")
    dpo = resolve_adapter(
        args.dpo_adapter,
        [
            PROJECT_ROOT / "outputs/dpo_adapter_v3",
            PROJECT_ROOT / "outputs/dpo_adapter_v2/dpo_policy",
            PROJECT_ROOT / "outputs/dpo_adapter_v2",
            PROJECT_ROOT / "outputs/dpo_adapter",
        ],
        "DPO",
    )
    manager = ModelManager(sft, dpo, args.max_seq_length)
    app = build_ui(manager)
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        show_error=True,
    )


if __name__ == "__main__":
    main()
