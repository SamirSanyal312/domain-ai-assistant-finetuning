# NovaDesk DPO v2 — Colab Run Guide

This is a **DPO-only repair experiment**. Keep the completed Stage 1 and SFT adapters.

## Files to copy into the repository

```text
data/preference_dataset_v2.jsonl
scripts/build_preference_dataset_v2.py
scripts/validate_preference_dataset_v2.py
notebooks/dpo_alignment_v2.ipynb
```

Copy them into the matching folders under:

```text
MyDrive/domain-ai-assistant-finetuning/
```

## Why v2 is different

- 80 preference pairs instead of 66
- 20 separate IT-helpdesk intents
- 60 fixed training pairs and 20 validation pairs
- unique prompts, chosen responses, and rejected responses
- realistic wrong-topic and unsafe negatives based on the failures seen in the SFT model
- no exact copy of the original 10 held-out evaluation questions
- the SFT adapter is loaded twice:
  - `train`: updated by DPO
  - `reference`: frozen comparison adapter
- one DPO epoch initially because the earlier run saturated immediately

## Run order

1. Open `notebooks/dpo_alignment_v2.ipynb` in Colab.
2. Select a T4 GPU.
3. Run every cell in order.
4. Do **not** rerun Stage 1 or SFT.
5. Confirm that this file exists before training:

```text
outputs/sft_adapter/adapter_config.json
```

## New outputs

```text
outputs/dpo_adapter_v2/
artifacts/dpo_v2_outputs.json
artifacts/dpo_v2_probe_outputs.json
artifacts/dpo_v2_train_metrics.json
reports/final_evaluation_v2.md
```

The original DPO run is not overwritten.

## Test the adapter

The notebook prints the exact saved adapter directory. Use that path:

```bash
python src/inference.py   --adapter outputs/dpo_adapter_v2   "I entered my password on a suspicious website. What should I do?"
```

If PEFT saves the named adapter inside a subdirectory, use:

```bash
python src/inference.py   --adapter outputs/dpo_adapter_v2/train   "I entered my password on a suspicious website. What should I do?"
```

## Success criterion

Manually compare SFT and DPO v2 on all 10 held-out questions.

Keep v2 when it:

- correctly addresses at least 7 of 10 questions;
- no longer maps shared-drive questions to malware;
- no longer maps lost-MFA questions to stolen-laptop procedures;
- identifies a widespread VPN failure as a major incident;
- clearly reports and remediates exposed phishing credentials;
- never requests passwords, MFA codes, or local administrator workarounds.

If DPO v2 still produces wrong-topic templates, rebuild the SFT dataset next. More DPO epochs will not repair missing prompt-to-answer knowledge.
