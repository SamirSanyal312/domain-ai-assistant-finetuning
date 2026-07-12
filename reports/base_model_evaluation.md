# Base Model Evaluation

This report is populated by `notebooks/non_instruction_finetuning.ipynb` before Stage 1 training. The notebook asks the same ten questions later used for SFT and DPO evaluation and saves raw outputs in `artifacts/base_outputs.json`.

| # | Question | Base model answer | Problem observed |
|---|---|---|---|
| 1 | My account is locked and keeps locking again after support unlocks it. What should I do? | Run the Stage 1 notebook | Assess genericness, policy mismatch, missing safety guidance, or unsupported claims |
| 2 | I clicked a suspicious email link and entered my corporate password. What actions should I take now? | Run the Stage 1 notebook | Assess genericness, policy mismatch, missing safety guidance, or unsupported claims |
| 3 | The VPN gives an error and will not connect from home. What should I check and include in the ticket? | Run the Stage 1 notebook | Assess genericness, policy mismatch, missing safety guidance, or unsupported claims |
| 4 | My company laptop was stolen from my car. Can I wait until tomorrow before reporting it? | Run the Stage 1 notebook | Assess genericness, policy mismatch, missing safety guidance, or unsupported claims |
| 5 | I need software that is not listed in Company Portal. How do I get it installed? | Run the Stage 1 notebook | Assess genericness, policy mismatch, missing safety guidance, or unsupported claims |
| 6 | Half of the company cannot connect to VPN. How should this be prioritized? | Run the Stage 1 notebook | Assess genericness, policy mismatch, missing safety guidance, or unsupported claims |
| 7 | I lost the phone used for MFA and have no backup factor. How can I regain access? | Run the Stage 1 notebook | Assess genericness, policy mismatch, missing safety guidance, or unsupported claims |
| 8 | A coworker has access to a shared drive. Can the help desk copy their permissions to me? | Run the Stage 1 notebook | Assess genericness, policy mismatch, missing safety guidance, or unsupported claims |
| 9 | My laptop is very slow. Is it okay to disable endpoint security to improve performance? | Run the Stage 1 notebook | Assess genericness, policy mismatch, missing safety guidance, or unsupported claims |
| 10 | A support technician asks me to send my password and approve an unexpected MFA prompt. What should I do? | Run the Stage 1 notebook | Assess genericness, policy mismatch, missing safety guidance, or unsupported claims |

Do not replace these cells with invented results. Run the notebook and paste or programmatically generate the actual answers.
