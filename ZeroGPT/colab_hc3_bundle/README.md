# HC3 Colab Project

This folder is meant to be uploaded to Google Drive, while the notebook file should be uploaded into Colab separately.

It needs to include:

- full HC3 data
- duplicated `gpt_zero/` code
- a local `gpt2` model under `hf_models/gpt2/` (not included on git due to size; run the optional download cell in `hc3_colab_workflow.ipynb` once to recreate it)
- trained SVM and XGBoost baseline models
- GPTZero-like training code
- reference metrics
- a ready-to-run notebook

## Main File

Open [hc3_colab_workflow.ipynb](./hc3_colab_workflow.ipynb).

## Quick Start

1. Upload `hc3_colab_workflow.ipynb` into Colab directly.
2. Select a GPU runtime in Colab.
   Runtime > Change runtime type > GPU
3. Before uploading the folder, strongly prefer adding the scorer model under `hf_models/gpt2/`.
   Run the optional download cell in `hc3_colab_workflow.ipynb` once with `REPO_DIR` pointing at this folder to populate `hf_models/gpt2/` locally before uploading to Google Drive.
4. Upload this folder to your Google Drive.
5. Mount Google Drive in the notebook.
6. Set `REPO_DIR` to the folder path in Drive.
7. Run the cells in order.

Do not run the notebook from Google Drive. Upload the notebook into Colab itself and use the Drive folder only as the project/data path.

If `hf_models/gpt2/` is present, the notebook will use that local model. If it is missing, the GPTZero section will fall back to downloading `gpt2` from Hugging Face at runtime.

Recommended `REPO_DIR`:

```python
REPO_DIR = "/content/drive/MyDrive/colab_hc3_bundle"
```

## What You Can Run

Baseline models:

- use the existing SVM and XGBoost models
- or retrain them in Colab to overwrite `artifacts/models/baselines/` and refresh `artifacts/runs/hc3_baselines_run/`

GPTZero-like model:

- train it in Colab on HC3
- score the test split
- generate evaluation metrics and comparison plots
- overwrite `artifacts/models/gptzero_like/` and refresh `artifacts/runs/hc3_gptzero_run/` when you rerun training
- prefer the local `hf_models/gpt2/` copy when present
- fall back to downloading `gpt2` only if the local folder is missing

## Input Format

The code accepts split files named:

- `train.parquet`, `val.parquet`, `test.parquet`
- `train.csv`, `val.csv`, `test.csv`

Parquet is preferred for large runs, but CSV is supported.

## Outputs

Baseline outputs go under:

- `artifacts/runs/hc3_baselines_run/`

GPTZero-like runs go under:

- `artifacts/runs/hc3_gptzero_run/`

Each run writes:

- `predictions/`
- `metrics/`
- `run_config.json`

## Notes

- HC3 is already prepared in `artifacts/data/hc3/`.
- The local GPTZero scorer model is not included on git; run the notebook's optional download cell once to recreate `hf_models/gpt2/` when needed.
- The baseline models are already available in `artifacts/models/baselines/`; retraining in the notebook overwrites that same directory instead of creating a parallel baseline model path.
- The GPTZero-like model is intended to be trained in Colab, and rerunning that section overwrites the saved detector artifacts in `artifacts/models/gptzero_like/` and refreshes `artifacts/runs/hc3_gptzero_run/`.
- The recommended workflow is to populate `hf_models/gpt2/` locally before uploading the folder to Drive, so Colab does not need to download the scorer model at runtime.
- If you choose to push `hf_models/gpt2/pytorch_model.bin` to GitHub, use Git LFS.
- If you previously ran the install cell before these pinned versions, restart the Colab runtime and run the install cell again.
- If baseline model loading fails after a version mismatch, rerun the dependency install cell from a fresh Colab runtime or retrain the baselines in Colab.
- If GPTZero-like is too slow, try `distilgpt2` or reduce `max_sentences_per_text`.
- If XGBoost GPU mode fails, change `xgb_device` to `"cpu"`.
