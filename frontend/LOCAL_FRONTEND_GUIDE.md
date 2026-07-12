# NovaDesk Local Frontend

This Gradio application exposes three modes from one locally loaded model:

1. **Base Model** — Qwen2.5-1.5B-Instruct with adapters disabled
2. **SFT Model** — the supervised NovaDesk adapter
3. **DPO Model** — the newest NovaDesk DPO adapter

Only one quantized base model is loaded. The two LoRA adapters are switched at inference time.

## Copy into the project

Copy the `frontend/` folder into the root of `domain-ai-assistant-finetuning`.

## Copy model adapters from Google Drive

The model weights were ignored by Git, so download these folders from Google Drive:

```text
MyDrive/domain-ai-assistant-finetuning/outputs/sft_adapter/
MyDrive/domain-ai-assistant-finetuning/outputs/dpo_adapter_v2/
```

Place them under the local repository's `outputs/` directory. The DPO adapter will commonly be:

```text
outputs/dpo_adapter_v2/dpo_policy/adapter_config.json
```

The application searches automatically for:

```text
outputs/sft_adapter_v2/
outputs/sft_adapter/
outputs/dpo_adapter_v3/
outputs/dpo_adapter_v2/dpo_policy/
outputs/dpo_adapter_v2/
outputs/dpo_adapter/
```

## Install and run on Windows

A supported CUDA GPU is strongly recommended.

```powershell
nvidia-smi
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install unsloth
pip install -r frontend\requirements-frontend.txt
python frontend\app.py --inbrowser
```

Open:

```text
http://127.0.0.1:7860
```

The first run downloads the base model from Hugging Face unless it is already cached.

## Explicit adapter paths

```powershell
python frontend\app.py `
  --sft-adapter "D:\models\novadesk\sft_adapter" `
  --dpo-adapter "D:\models\novadesk\dpo_adapter_v2\dpo_policy" `
  --inbrowser
```

## Fair comparison settings

Use:

```text
Temperature: 0
Top-p: 0.9
Maximum new tokens: 220
```

Ask the same question in each mode. The app clears the chat when the mode changes so earlier answers do not contaminate the comparison.

## Troubleshooting

**Adapter not found:** verify that the folder contains `adapter_config.json` and `adapter_model.safetensors`.

**Port in use:**

```powershell
python frontend\app.py --port 7861 --inbrowser
```

**GPU memory issue:**

```powershell
python frontend\app.py --max-seq-length 1024 --inbrowser
```

**Temporary public link:** use `--share`, but do not enter confidential data.
