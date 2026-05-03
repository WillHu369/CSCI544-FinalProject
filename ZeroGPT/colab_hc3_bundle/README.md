# HC3 Colab Workflow

This folder is the evaluator-facing Colab bundle for the three classical baselines and the GPTZero-like detector.

## What To Upload

Upload this entire folder to Google Drive:

```text
ZeroGPT/colab_hc3_bundle/
```

Open this notebook directly in Colab:

```text
hc3_colab_workflow.ipynb
```

Do not run the notebook from Drive. Keep the project folder in Drive and the notebook in Colab.

## Quick Run

1. Upload `ZeroGPT/colab_hc3_bundle` to Google Drive.
2. Open `hc3_colab_workflow.ipynb` in Colab.
3. Select a GPU runtime.
4. Set: pythonREPO_DIR = "/content/drive/MyDrive/colab_hc3_bundle"
5. Run the notebook cells in order.

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

