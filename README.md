# NovaDesk IT Helpdesk Assistant — Three-Stage Fine-Tuning with Unsloth

A GitHub-ready implementation of a domain-specific IT helpdesk assistant trained in three stages:

1. Non-instruction domain adaptation on raw IT policy text
2. Supervised instruction fine-tuning (SFT)
3. Direct Preference Optimization (DPO)

> **Important:** The repository includes complete datasets, executable Colab notebooks, evaluation automation, and inference code. The committed Markdown evaluation files are templates until the GPU notebooks are run. Do not present placeholder text as measured results.

## Domain and business problem

The fictional company **NovaDesk** needs an internal assistant that gives consistent first-line IT support. The assistant should:

- troubleshoot account, VPN, network, email, device, printer, and software issues;
- collect useful diagnostics without asking for secrets;
- follow least privilege and sensitive-data rules;
- recognize phishing, malware, lost devices, and widespread outages;
- explain when to stop self-service troubleshooting and escalate.

IT helpdesk was selected instead of healthcare because a helpdesk assistant has lower safety risk, can be evaluated against a clear fictional policy, and has a relevant public ticket dataset for taxonomy and augmentation.

## Dataset details

| File | Count | Purpose |
|---|---:|---|
| `data/non_instruction_data.txt` | 66 paragraphs | Domain language and policy adaptation |
| `data/instruction_dataset.jsonl` | 132 pairs | User-question → policy answer SFT |
| `data/preference_dataset.jsonl` | 66 triples | Chosen vs rejected DPO alignment |
| `data/evaluation_questions.json` | 10 questions | Fixed before/after evaluation |
| `data/public_ticket_sample.jsonl` | 50 rows | Optional exploratory public sample |

The primary training data is synthetic but is grounded in one consistent fictional knowledge base. Run `python scripts/validate_data.py` to verify minimum counts, schema, duplicates, and basic safety checks.

### Public dataset reference

The optional augmentation source is `Tobi-Bueck/customer-support-tickets` on Hugging Face. Its dataset card lists English and German support tickets, 61.8K rows, and a CC BY-NC 4.0 license. See `data/DATASET_CARD.md` before reuse. The required training notebooks do not depend on this public sample.

## Base model

`unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`

The underlying Qwen2.5 1.5B Instruct model is small enough for a Colab GPU and includes a chat template, making the same unmodified checkpoint usable for the baseline evaluation and all three training stages. The Unsloth checkpoint is pre-quantized for memory-efficient 4-bit loading.

## Repository structure

```text
domain-ai-assistant-finetuning/
├── data/
│   ├── non_instruction_data.txt
│   ├── instruction_dataset.jsonl
│   ├── preference_dataset.jsonl
│   ├── evaluation_questions.json
│   ├── public_ticket_sample.jsonl
│   ├── DATASET_CARD.md
├── notebooks/
│   ├── non_instruction_finetuning.ipynb
│   ├── instruction_finetuning.ipynb
│   └── dpo_alignment.ipynb
├── reports/
│   ├── base_model_evaluation.md
│   ├── sft_model_comparison.md
│   ├── final_evaluation.md
│   └── fine_tuning_explanation.md
├── scripts/
│   ├── prepare_public_dataset.py
│   └── validate_data.py
├── src/
│   ├── common.py
│   └── inference.py
├── requirements.txt
└── README.md
```

## Training workflow

### 1. Validate the repository data

```bash
python scripts/validate_data.py
```

Expected counts:

```text
raw_paragraphs: 66
instruction_examples: 132
preference_examples: 66
```

### 2. Create a GitHub repository

```bash
git init
git add .
git commit -m "Initial NovaDesk fine-tuning project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/domain-ai-assistant-finetuning.git
git push -u origin main
```

### 3. Run notebooks in Google Colab

Use a GPU runtime and run in this order:

1. `notebooks/non_instruction_finetuning.ipynb`
2. `notebooks/instruction_finetuning.ipynb`
3. `notebooks/dpo_alignment.ipynb`

Clone the repository in Colab before opening/running a notebook:

```python
!git clone https://github.com/YOUR_USERNAME/domain-ai-assistant-finetuning.git
%cd domain-ai-assistant-finetuning
```

Each notebook saves the next-stage adapter under `outputs/`. The Stage 1 notebook saves base answers; the SFT and DPO notebooks automatically generate comparison reports from actual model outputs.

### 4. Preserve results

Colab storage is temporary. Download these folders after each run or copy them to Google Drive:

- `outputs/non_instruction_adapter`
- `outputs/sft_adapter`
- `outputs/dpo_adapter`
- `artifacts/*.json`
- updated `reports/*.md`

Commit the updated reports and selected training screenshots/logs. Avoid committing large checkpoints unless Git LFS is configured; LoRA adapters are smaller but can still be large.

## LoRA / QLoRA configuration

| Parameter | Value |
|---|---|
| 4-bit base model | enabled |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.0 |
| Target modules | q/k/v/o projections and gate/up/down MLP projections |
| Gradient checkpointing | Unsloth |
| Maximum sequence length | 1,024 |
| Stage 1 learning rate | 2e-4 |
| SFT learning rate | 1e-4 |
| DPO learning rate | 1e-5 |
| DPO beta | 0.1 |

## Inference

After the DPO adapter exists:

```bash
python src/inference.py "I clicked a phishing link and entered my password. What should I do?"
```

Interactive mode:

```bash
python src/inference.py
```

## Evaluation approach

All three stages use the same ten questions. The notebooks preserve raw outputs and build Markdown tables. Human review is still required for:

- correctness;
- NovaDesk policy accuracy;
- clarity and helpfulness;
- credential and data safety;
- escalation behavior;
- professional tone;
- hallucination reduction.

## Expected behavior changes

These are hypotheses to verify, not claimed measurements:

- Base model: generally useful but may be generic or omit NovaDesk-specific rules.
- SFT model: should cite the 15-minute lockout wait, approved portals, required ticket details, and explicit escalation conditions.
- DPO model: should more consistently reject password/code sharing, unsupported restoration promises, unsafe malware cleanup, and delayed lost-device reporting.

## Training screenshots or logs

Before final submission, add screenshots showing:

1. GPU runtime and installed versions
2. dataset counts
3. trainable parameter count
4. Stage 1 loss
5. SFT loss/evaluation metrics
6. DPO chosen/rejected reward or margin metrics
7. one before-vs-after answer

Place images in `reports/images/` and embed them here.

## Challenges

- Small datasets can overfit; validation loss and held-out questions must be reviewed.
- Public support-ticket data can be generic, noisy, duplicated, or inconsistent with a company policy.
- DPO pairs must differ meaningfully; trivial wording changes teach little.
- A 1.5B model may memorize policy phrases but still make mistakes outside the curated topics.
- Colab sessions are temporary, so adapters and reports must be saved promptly.

## Future improvements

- Expand and diversify paraphrases while keeping a policy source of truth.
- Add retrieval-augmented generation over current IT documentation.
- Add automated safety checks for secrets, unsafe commands, and false outage promises.
- Evaluate on unseen adversarial prompts and multi-turn conversations.
- Track exact-match policy facts and human preference scores.
- Deploy a merged or GGUF model only after license, privacy, and security review.

## License

Code is provided under the MIT License. The fictional NovaDesk training data is included for educational use. The optional public sample retains its original CC BY-NC 4.0 attribution and restrictions; see `data/DATASET_CARD.md`.
