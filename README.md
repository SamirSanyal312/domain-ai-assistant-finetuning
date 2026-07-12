# NovaDesk IT Helpdesk Assistant

A domain-specific IT helpdesk assistant fine-tuned with **Unsloth**, **QLoRA**, **Supervised Fine-Tuning (SFT)**, and **Direct Preference Optimization (DPO)**.

This project implements the complete three-stage workflow required by the assignment:

```text
Qwen2.5-1.5B-Instruct
        ↓
Stage 1: Non-instruction domain adaptation
        ↓
Stage 2: Instruction fine-tuning (SFT)
        ↓
Stage 3: Preference alignment (DPO)
        ↓
NovaDesk IT Helpdesk Assistant
```

## Project status

| Component | Status |
|---|---|
| Raw-domain dataset | Completed |
| Non-instruction fine-tuning | Completed |
| Instruction dataset | Completed |
| Instruction fine-tuning | Completed |
| Preference dataset | Completed |
| DPO alignment | Completed |
| Base/SFT/DPO inference outputs | Completed |
| Interactive inference script | Completed |
| Human evaluation reports | Generated; judgment columns require final manual review |

> **Important result:** All three training stages executed successfully, but manual testing showed mixed final response quality. The project demonstrates that successful training, low loss, and high DPO reward accuracy do not automatically guarantee a reliable assistant when the dataset is small or repetitive.

---

## Domain and business problem

The selected domain is **IT Helpdesk Support**.

The fictional company **NovaDesk** needs an internal assistant that can:

- troubleshoot account, VPN, network, email, device, printer, and software issues;
- collect useful ticket information without requesting passwords or MFA codes;
- provide safe first-line support;
- recognize phishing, malware, lost-device, and outage scenarios;
- follow least-privilege and approved-software practices;
- identify when the user must stop troubleshooting and escalate to the Service Desk or Security Operations.

IT helpdesk was selected because it has clear operational procedures, measurable support scenarios, and relevant public ticket datasets.

---

## Model

The project uses:

```text
unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
```

The underlying Qwen2.5 model is an instruction-tuned causal language model with approximately **1.54 billion parameters**. The 4-bit Unsloth checkpoint makes adapter training practical on a Google Colab Tesla T4 GPU.

### Why this model was selected

- small enough for limited Colab VRAM;
- supports conversational prompts and chat templates;
- suitable for LoRA/QLoRA fine-tuning;
- capable of baseline instruction following;
- recommended model size for the assignment.

---

## Dataset details

| File | Size | Purpose |
|---|---:|---|
| `data/non_instruction_data.txt` | 66 paragraphs | Domain terminology and policy adaptation |
| `data/instruction_dataset.jsonl` | 132 examples | Question-to-answer supervised fine-tuning |
| `data/preference_dataset.jsonl` | 66 examples | Chosen-versus-rejected DPO alignment |
| `data/evaluation_questions.json` | 10 questions | Fixed before/after evaluation |
| `data/public_ticket_sample.jsonl` | 50 rows | Optional public-data exploration |

The project exceeds the assignment minimums of:

- 50 raw-domain paragraphs;
- 100 instruction-response examples;
- 50 preference examples;
- 10 evaluation questions.

### Dataset strategy

The primary datasets use a fictional but consistent NovaDesk policy. This avoids exposing real company information and provides a controlled source of truth for evaluation.

An optional sample derived from the public `Tobi-Bueck/customer-support-tickets` dataset is included for taxonomy exploration and future augmentation. The required training notebooks do not depend on this optional sample. Review `data/DATASET_CARD.md` before reusing public data.

### Validate the data

```bash
python scripts/validate_data.py
```

Expected counts:

```text
raw_paragraphs: 66
instruction_examples: 132
preference_examples: 66
evaluation_questions: 10
```

---

## Repository structure

```text
domain-ai-assistant-finetuning/
├── artifacts/
│   ├── base_outputs.json
│   ├── non_instruction_outputs.json
│   ├── sft_outputs.json
│   ├── dpo_outputs.json
│   └── *_train_metrics.json
├── data/
│   ├── non_instruction_data.txt
│   ├── instruction_dataset.jsonl
│   ├── preference_dataset.jsonl
│   ├── evaluation_questions.json
│   ├── public_ticket_sample.jsonl
│   └── DATASET_CARD.md
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
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

The large `outputs/` directory, model checkpoints, tokenizer caches, and compiled Unsloth caches are intentionally excluded from GitHub.

---

# Training workflow

## Stage 1: Non-instruction fine-tuning

The first stage performs continued language-model training on raw NovaDesk policy text.

### Goal

Teach the model:

- IT-helpdesk terminology;
- NovaDesk escalation language;
- common ticket details;
- password and MFA safety rules;
- lost-device and phishing terminology;
- approved support practices.

This stage uses next-token prediction rather than question-answer examples.

### Configuration

| Parameter | Value |
|---|---:|
| Quantization | 4-bit |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.0 |
| Micro-batch size | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Epochs | 3 |
| Learning rate | `2e-4` |
| Maximum sequence length | 1,024 |

Target modules:

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

### Measured Stage 1 results

| Epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 4.2282 | 4.3969 |
| 2 | 4.2283 | 4.0923 |
| 3 | 3.9348 | 3.9551 |

The adapter was saved under:

```text
outputs/non_instruction_adapter/
```

### Observation

The model showed greater familiarity with IT terminology after this stage, but its answers remained generic. This was expected because raw-text adaptation does not strongly teach question-answer behavior.

---

## Stage 2: Supervised instruction fine-tuning

The second stage continues from the Stage 1 adapter and trains on 132 question-answer examples.

### Goal

Teach the model how to convert an employee’s IT problem into a concise, safe, domain-specific response.

### Dataset split

| Split | Examples |
|---|---:|
| Training | 118 |
| Validation | 14 |

### Configuration

| Parameter | Value |
|---|---:|
| Trainable parameters | 18,464,768 |
| Total parameters | 1,562,179,072 |
| Trainable percentage | 1.182% |
| Micro-batch size | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Epochs | 4 |
| Optimizer steps | 60 |
| Learning rate | `1e-4` |
| Completion-only loss | Enabled |

### Measured SFT results

| Epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 2.6651 | 2.4403 |
| 2 | 1.7636 | 1.9472 |
| 3 | 1.3246 | 1.7089 |
| 4 | 1.1482 | 1.6681 |

Average training loss:

```text
1.8923
```

The adapter was saved under:

```text
outputs/sft_adapter/
```

### Observation

SFT produced stronger company-oriented answers for some scenarios, including:

- stolen-company-device reporting;
- individual VPN ticket details;
- approved software requests;
- password and unexpected MFA prompt refusal.

However, some responses were mapped to the wrong support topic. Examples included a shared-drive access question receiving malware advice and an MFA-recovery question receiving stolen-device guidance.

---

## Stage 3: Direct Preference Optimization

DPO trains the SFT model using preference triples:

```json
{
  "prompt": "User question",
  "chosen": "Preferred safe and helpful response",
  "rejected": "Weaker, unsafe, incomplete, or generic response"
}
```

DPO directly increases the relative preference for the chosen completion over the rejected completion compared with a reference policy. It does not require a separately trained reward model.

### Dataset split

| Split | Examples |
|---|---:|
| Training | 59 |
| Validation | 7 |

### Configuration

| Parameter | Value |
|---|---:|
| Beta | 0.1 |
| Loss type | Sigmoid |
| Micro-batch size | 1 |
| Gradient accumulation | 8 |
| Effective batch size | 8 |
| Epochs | 2 |
| Optimizer steps | 16 |
| Learning rate | `5e-6` |
| Padding side | Left |

### Measured DPO results

| Epoch | DPO loss | Chosen reward | Rejected reward | Reward accuracy | Reward margin |
|---:|---:|---:|---:|---:|---:|
| 1 | ~0.000000 | 15.8113 | -3.4884 | 1.0000 | 19.2997 |
| 2 | ~0.000001 | 15.8253 | -3.6390 | 1.0000 | 19.4643 |

Final recorded training loss:

```text
4.8154e-7
```

The adapter was saved under:

```text
outputs/dpo_adapter/
```

### Interpreting the metrics

- `rewards/chosen` measures the implicit reward assigned to preferred responses.
- `rewards/rejected` measures the implicit reward assigned to rejected responses.
- `rewards/accuracies` measures how often the chosen response scores higher.
- `rewards/margins` measures the average separation between chosen and rejected rewards.

The perfect reward accuracy indicates that the model separated the supplied preference pairs. It does **not** prove perfect real-world behavior.

---

# Evaluation

The same 10 questions were used for:

1. the base model;
2. the SFT model;
3. the DPO-aligned model.

Evaluation criteria:

- correctness;
- domain accuracy;
- safety;
- helpfulness;
- clarity;
- professional tone;
- escalation behavior;
- hallucination risk.

Detailed outputs are stored in:

```text
reports/base_model_evaluation.md
reports/sft_model_comparison.md
reports/final_evaluation.md
```

## Honest final observation

Manual review found **mixed performance**.

### Successful behaviors

The fine-tuned model often improved:

- immediate stolen-device escalation;
- collection of useful VPN ticket details;
- refusal to accept passwords and one-time codes;
- use of approved software-request processes;
- professional helpdesk wording.

### Weak behaviors

The model sometimes:

- selected a memorized answer from the wrong IT category;
- repeated nearly identical SFT and DPO answers;
- omitted the immediate password-change and incident-reporting actions after phishing;
- treated a company-wide VPN outage as an individual troubleshooting problem;
- gave unrelated malware advice for a permissions request;
- suggested unsafe or unrealistic actions such as local administrator access or rebooting a domain controller.

### Why this happened

The primary limitation was dataset quality rather than pipeline execution:

- multiple instructions reused the same target response;
- the SFT dataset had limited answer diversity;
- several DPO rejected answers were too easy to distinguish;
- DPO chosen answers frequently repeated the existing SFT targets;
- few examples covered identity questions, malicious requests, out-of-domain prompts, and nuanced incident-priority decisions;
- the 1.5B model has limited capacity;
- greedy decoding makes memorized templates more visible.

The key lesson is:

> Low loss and perfect preference accuracy measure performance on the supplied training objective. Human evaluation is still necessary to determine whether responses are actually correct, relevant, and safe.

---

# Inference

The final script loads the 4-bit base model and the saved DPO LoRA adapter.

## Single question

```bash
python src/inference.py \
  "My company laptop was stolen. Can I report it tomorrow?"
```

## Interactive mode

```bash
python src/inference.py
```

Example:

```text
Loading NovaDesk model...
Ready. Type 'exit' to stop.

You: Can I send you my password?

Assistant:
Do not share passwords, one-time codes, or private keys...
```

The model is loaded once in interactive mode, making repeated questions faster.

> A CUDA-capable environment is recommended because the model is loaded in 4-bit mode.

---

# Running the project in Google Colab

Run the notebooks in this exact order:

```text
1. notebooks/non_instruction_finetuning.ipynb
2. notebooks/instruction_finetuning.ipynb
3. notebooks/dpo_alignment.ipynb
```

Use a GPU runtime:

```text
Runtime → Change runtime type → T4 GPU
```

The project was stored in Google Drive so adapters and generated reports persisted between Colab sessions.

Example setup:

```python
from google.colab import drive
drive.mount("/content/drive")

PROJECT = "/content/drive/MyDrive/domain-ai-assistant-finetuning"

!rm -rf /content/domain-ai-assistant-finetuning
!ln -s "{PROJECT}" /content/domain-ai-assistant-finetuning

%cd /content/domain-ai-assistant-finetuning
```

Unsloth must be imported before TRL, Transformers, and PEFT:

```python
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
```

For Qwen2.5, the project explicitly aligns the tokenizer with:

```python
tokenizer.eos_token = "<|im_end|>"
tokenizer.pad_token = "<|endoftext|>"
```

---

# Reports

The notebooks automatically generated the model-answer columns in the Markdown reports.

The following fields still require human judgment before final submission:

- `Problem observed`
- `Which is better?`
- `Best answer`
- `Reason`

The evaluator should not automatically select the DPO response. The answer should be chosen based on the actual content.

---

# Challenges faced

1. **Qwen EOS-token mismatch**  
   TRL initially received the invalid placeholder `<EOS_TOKEN>`. This was resolved by importing Unsloth before TRL and explicitly setting Qwen’s `<|im_end|>` EOS token.

2. **Colab persistence**  
   Google Drive was used because Colab’s local filesystem is temporary.

3. **Verbose inference output**  
   The inference script was updated to suppress non-fatal Transformers and Unsloth startup messages.

4. **Dataset repetition**  
   Repeated target answers caused the model to behave like a template retriever.

5. **Misleadingly strong DPO metrics**  
   Reward accuracy reached 1.0 because many rejected answers were very weak. Human evaluation showed that this did not translate into universal response improvement.

6. **Small-model limitations**  
   A 1.5B model is practical for Colab but has limited robustness on unseen or ambiguous prompts.

---

# Future improvements

- Create unique, question-specific SFT answers.
- Increase the SFT dataset to at least 250–500 diverse examples.
- Build DPO rejected answers from real SFT failures rather than only obviously bad answers.
- Add hard preference pairs where both answers sound plausible.
- Add identity, refusal, malicious-request, access-control, MFA-recovery, and major-incident examples.
- Add unseen and adversarial evaluation prompts.
- Use a separate frozen SFT reference adapter during DPO.
- Evaluate stochastic and deterministic decoding separately.
- Add retrieval-augmented generation over current IT policy documents.
- Test a larger 3B–7B model when more GPU memory is available.
- Add multi-turn conversation memory.
- Introduce automatic policy and secret-leak checks.

---

# GitHub repository hygiene

Large generated files are excluded through `.gitignore`:

```gitignore
outputs/
*.safetensors
checkpoint-*/
huggingface_tokenizers_cache/
unsloth_compiled_cache/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
```

The repository should include:

- datasets;
- executed notebooks;
- reports;
- small JSON artifacts;
- source code;
- screenshots;
- README and license.

It should not include:

- model checkpoints;
- LoRA `.safetensors` weights;
- Hugging Face caches;
- compiled Unsloth caches;
- Python cache folders.

---

# Training evidence

Add your screenshots under:

```text
reports/images/
```

Recommended filenames:

```text
01_dataset_validation.png
02_non_instruction_training.png
03_sft_training.png
04_dpo_training.png
05_model_comparison.png
06_final_inference.png
```

Example Markdown:

```markdown
![SFT training](reports/images/03_sft_training.png)
![DPO training](reports/images/04_dpo_training.png)
![Final inference](reports/images/06_final_inference.png)
```

---

# Key learning outcomes

This project demonstrates:

- raw-text domain adaptation;
- 4-bit QLoRA on limited hardware;
- supervised prompt-completion training;
- preference data preparation;
- DPO alignment;
- adapter saving and loading;
- before-versus-after evaluation;
- interactive inference;
- the importance of dataset diversity and human evaluation.

A concise interview explanation:

> I built a domain-specific IT helpdesk assistant using Unsloth and Qwen2.5-1.5B. I first performed non-instruction domain adaptation on raw policy text, continued with supervised instruction fine-tuning on question-answer pairs, and then applied DPO using chosen and rejected responses. I evaluated the base, SFT, and DPO models on the same held-out questions. The pipeline completed successfully, but human evaluation revealed that repetitive training data caused topic-mismatched responses, demonstrating why dataset quality matters more than training loss alone.

---

# References

- [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [Unsloth Qwen2.5 1.5B 4-bit checkpoint](https://huggingface.co/unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit)
- [Hugging Face TRL DPO Trainer](https://huggingface.co/docs/trl/en/dpo_trainer)
- [Unsloth preference optimization guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto)
- [Direct Preference Optimization paper](https://arxiv.org/abs/2305.18290)
- [Optional public support-ticket dataset](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)

---

# License

The project code is released under the **MIT License**.

The NovaDesk data is fictional and included for educational use. Any optional public dataset content remains subject to its original dataset license and attribution requirements.
