# Dataset sources and licensing

## Primary training data

The three required training files in this repository are a curated synthetic dataset for a fictional company, **NovaDesk**. They were generated from one internally consistent knowledge base and then checked for contradictions, unsafe credential-sharing guidance, and duplicated prompts. This avoids representing fictional policies as the policy of a real organization.

## Public helpdesk dataset used for taxonomy review

- Dataset: `Tobi-Bueck/customer-support-tickets`
- Host: Hugging Face Datasets
- License listed on the dataset card: **CC BY-NC 4.0**
- Dataset DOI listed on the card: `10.57967/hf/6184`
- Repository use: `data/public_ticket_sample.jsonl` contains a small attributed sample for exploratory analysis only. It is not required by the training notebooks.
- Noncommercial restriction: review the license before any commercial reuse.

The optional script `scripts/prepare_public_dataset.py` shows how to download, filter, sanitize, and convert English IT-support records. The assignment's required `instruction_dataset.jsonl` remains the smaller policy-grounded curated set so that evaluation has a clear source of truth.
