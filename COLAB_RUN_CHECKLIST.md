# Colab execution checklist

1. Create a new public or private GitHub repository and push this folder.
2. In Colab, select **Runtime → Change runtime type → T4 GPU**.
3. Clone the repository.
4. Run notebooks in the required order.
5. After each notebook, download the newly created adapter and artifacts.
6. Review generated Markdown reports manually.
7. Add training screenshots to `reports/images/` and README.
8. Commit the reports and screenshots.
9. Submit the GitHub repository URL.

## Recommended evidence

- `nvidia-smi` output
- dataset validation output
- LoRA trainable parameter count
- loss curve or trainer log
- base/SFT/DPO answer comparison
- final inference script output

## Honesty rule

Do not claim the adapters were trained or that DPO improved the model until the notebooks have actually run and the outputs have been reviewed.
