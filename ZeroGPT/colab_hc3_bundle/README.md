# HC3 Colab Workflow

This folder is the evaluator-facing Colab bundle for the three classical baselines and the GPTZero-like detector.

## Environment setup

- Google Drive
- Google Colab
- a Colab GPU runtime

Upload this entire folder to Google Drive:

```text
ZeroGPT/colab_hc3_bundle/
```

Open this notebook directly in Colab:

```text
hc3_colab_workflow.ipynb
```

The notebook installs its runtime dependencies when you run the setup cells.

## Device / system used

- Platform: Google Colab
- Runtime: GPU runtime
- Storage: Google Drive for the uploaded `colab_hc3_bundle` folder

## How to run

Do not run the notebook from Drive. Keep the project folder in Drive and the notebook in Colab.

1. Upload `ZeroGPT/colab_hc3_bundle` to Google Drive.
1. Open `hc3_colab_workflow.ipynb` in Colab.
1. Select a GPU runtime.
1. Set:

```python
REPO_DIR = "/content/drive/MyDrive/colab_hc3_bundle"
```

1. Run the notebook cells in order.

## Optional Local GPT-2 Cache

If `hf_models/gpt2/` exists, the notebook uses it.

If it does not exist, the notebook can still run by downloading `gpt2` from Hugging Face at runtime.

## Test Dataset Evaluation

The notebook evaluates the kept dataset variants under:

```text
test_dataset/
```

That includes:

- original clean HC3 test set
- stylistic cleanup set
- recursive paraphrase depth 1/2/3 test sets
- T5 perturbation test set

Shared per-dataset metric exports are written to:

```text
metrics_share/
```

## How results are generated

The notebook uses the prepared data and test files in this folder to produce the reported metrics.

1. It trains or reuses the classical baselines and the GPTZero-like detector.
1. It evaluates the kept test sets: original clean, stylistic cleanup, recursive paraphrase depth 1/2/3, and T5 perturbation.
1. It writes baseline-run metrics under `artifacts/runs/hc3_baselines_run/metrics/`.
1. It writes shared per-dataset metric exports under `metrics_share/`.
