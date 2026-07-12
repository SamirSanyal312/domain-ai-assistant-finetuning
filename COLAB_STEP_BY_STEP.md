# Google Colab step-by-step execution guide

This project must be run in this order:

1. `notebooks/non_instruction_finetuning.ipynb`
2. `notebooks/instruction_finetuning.ipynb`
3. `notebooks/dpo_alignment.ipynb`

Use a Google Drive-backed repository folder so the adapter produced by one stage is available to the next stage.

## One-time preparation

1. Extract the project ZIP on your computer.
2. Create an empty GitHub repository named `domain-ai-assistant-finetuning`.
3. Push the extracted project folder to that repository.
4. In Google Drive, no manual upload is required; the first Colab setup cell clones the repository into `MyDrive`.

## At the beginning of Stage 1

Choose a GPU runtime, then insert and run this cell before the notebook's package-installation cell. Replace `YOUR_USERNAME`.

```python
from google.colab import drive
drive.mount('/content/drive')

PROJECT = '/content/drive/MyDrive/domain-ai-assistant-finetuning'
GITHUB_REPO = 'https://github.com/YOUR_USERNAME/domain-ai-assistant-finetuning.git'

import os
if not os.path.isdir(os.path.join(PROJECT, '.git')):
    !git clone "{GITHUB_REPO}" "{PROJECT}"

!rm -rf /content/domain-ai-assistant-finetuning
!ln -s "{PROJECT}" /content/domain-ai-assistant-finetuning
%cd /content/domain-ai-assistant-finetuning
```

Then run:

```python
!nvidia-smi
!python scripts/validate_data.py
```

Run every remaining Stage 1 cell in order. Save the notebook after training.

Expected Stage 1 outputs:

- `outputs/non_instruction_adapter/`
- `artifacts/base_outputs.json`
- `artifacts/non_instruction_outputs.json`
- `artifacts/non_instruction_train_metrics.json`
- updated `reports/base_model_evaluation.md`

## At the beginning of Stage 2 and Stage 3

Mount the same Drive folder and recreate the runtime symlink. Do not clone or delete the Drive project again.

```python
from google.colab import drive
drive.mount('/content/drive')

PROJECT = '/content/drive/MyDrive/domain-ai-assistant-finetuning'
!rm -rf /content/domain-ai-assistant-finetuning
!ln -s "{PROJECT}" /content/domain-ai-assistant-finetuning
%cd /content/domain-ai-assistant-finetuning
```

Run all notebook cells in chronological order.

Expected Stage 2 outputs:

- `outputs/sft_adapter/`
- `artifacts/sft_outputs.json`
- `artifacts/sft_train_metrics.json`
- updated `reports/sft_model_comparison.md`

Expected Stage 3 outputs:

- `outputs/dpo_adapter/`
- `artifacts/dpo_outputs.json`
- `artifacts/dpo_train_metrics.json`
- updated `reports/final_evaluation.md`

## Final verification

Run the final adapter through the inference script:

```python
!python src/inference.py "I clicked a phishing link and entered my password. What should I do?"
```

Manually review and complete the judgment columns in:

- `reports/base_model_evaluation.md`
- `reports/sft_model_comparison.md`
- `reports/final_evaluation.md`

Add screenshots to `reports/images/` showing the GPU, dataset validation, trainable parameter count, trainer loss, DPO reward metrics, and final inference response.

Do not commit `outputs/` or `artifacts/*.json`; they are intentionally ignored. Commit the executed notebooks, updated reports, screenshots, and README.
