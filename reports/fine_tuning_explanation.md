# Fine-Tuning Explanation

## Why full fine-tuning is expensive

A full fine-tune updates every model parameter. Even a 1.5-billion-parameter model needs memory not only for the weights, but also gradients, optimizer states, activations, and checkpoints. The optimizer states can use several times the memory of the model weights, so full fine-tuning is usually impractical on a free or small GPU.

## LoRA

Low-Rank Adaptation freezes the original model and inserts small trainable low-rank matrices into selected linear layers. Instead of updating the full attention and MLP weight matrices, training learns a compact change to them. This greatly reduces trainable parameters and makes adapters easy to save and share.

## QLoRA

QLoRA loads the frozen base model in 4-bit precision while training LoRA adapters in a higher compute precision. The compressed base weights reduce VRAM use, while the adapters learn the domain-specific changes. In this project, “QLoRA-style” means the model is loaded with `load_in_4bit=True` and only LoRA parameters are trainable.

## Why QLoRA helps on limited GPUs

Four-bit storage is much smaller than 16-bit storage, so a model that would not fit comfortably in a Colab T4 can often be trained with adapters. Gradient checkpointing and small micro-batches reduce activation memory further, at the cost of additional compute.

## Non-instruction fine-tuning

Non-instruction fine-tuning, also called continued pretraining or domain adaptation, trains next-token prediction on raw domain text. It helps the model become familiar with NovaDesk terminology, policies, escalation language, and troubleshooting style, but it does not by itself strongly teach question-answer behavior.

## Instruction fine-tuning (SFT)

SFT trains the model on explicit user-question and assistant-answer pairs. It teaches the assistant how to transform a helpdesk request into a clear, safe, policy-grounded response. The prompt-completion format lets training focus loss on the desired assistant response.

## Direct Preference Optimization (DPO)

DPO uses triples containing a prompt, a preferred answer, and a rejected answer. It directly increases the model's relative preference for chosen answers over rejected answers compared with a reference policy. It does not require training a separate reward model.

## SFT vs DPO

SFT teaches the model what answer to imitate. DPO teaches the model which of two plausible behaviors is preferable. Here, SFT learns NovaDesk answers; DPO reinforces safe, specific, professional responses over generic or credential-sharing responses.

## Configuration used

| Stage | Rank | Alpha | Dropout | Learning rate | Micro-batch | Gradient accumulation | Epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-instruction QLoRA | 16 | 32 | 0.0 | 2e-4 | 2 | 4 | 3 |
| Instruction SFT | continues same adapter | continues same adapter | 0.0 | 1e-4 | 2 | 4 | 4 |
| DPO | continues SFT adapter | continues SFT adapter | 0.0 | 5e-6 | 1 | 8 | 2 |

The target modules are `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. Maximum sequence length is 1,024 tokens. DPO beta is 0.1. These values are intentionally conservative for a small dataset and a limited Colab GPU; training curves and validation outputs should be checked for overfitting.

## Measured training results

### Non-instruction fine-tuning

| Epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 4.2282 | 4.3969 |
| 2 | 4.2283 | 4.0923 |
| 3 | 3.9348 | 3.9551 |

### Supervised fine-tuning

| Epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 2.6651 | 2.4403 |
| 2 | 1.7636 | 1.9472 |
| 3 | 1.3246 | 1.7089 |
| 4 | 1.1482 | 1.6681 |

The SFT stage completed 60 optimizer steps and reported an average training loss of approximately 1.8923.

### DPO alignment

| Epoch | Chosen reward | Rejected reward | Reward accuracy | Reward margin |
|---:|---:|---:|---:|---:|
| 1 | 15.8113 | -3.4884 | 1.0000 | 19.2997 |
| 2 | 15.8253 | -3.6390 | 1.0000 | 19.4643 |

The final DPO training loss was approximately `4.8154e-7`.

## Interpretation and lessons learned

The training pipeline completed successfully, but human evaluation showed that the DPO model did not consistently improve answer quality. A reward accuracy of 1.0 means that the chosen responses scored above the rejected responses for the supplied validation pairs; it does not mean that every generated answer is correct.

The main limitation was dataset quality. Several instruction prompts reused the same target response, causing the small model to memorize topic templates. Many DPO rejected answers were obviously weaker than the chosen answers, making preference separation too easy. Future datasets should contain unique question-specific responses, realistic rejected answers generated from actual SFT failures, and additional examples for identity, access control, incident priority, MFA recovery, refusal, and out-of-domain behavior.

## External references used for the safety review

- CISA, “Recognize and Report Phishing”: https://www.cisa.gov/secure-our-world/recognize-and-report-phishing
- CISA, “Avoiding Social Engineering and Phishing Attacks”: https://www.cisa.gov/news-events/news/avoiding-social-engineering-and-phishing-attacks
- Microsoft Entra, “Configure multifactor authentication settings — Report suspicious activity”: https://learn.microsoft.com/en-us/entra/identity/authentication/howto-mfa-mfasettings
- NIST CSRC, “Least Privilege”: https://csrc.nist.gov/glossary/term/least_privilege

