# DetectGPT Colab Setup Guide

This folder is designed to run end-to-end in **Google Colab** using files stored in **Google Drive**.

The notebook mounts your Drive, runs `run.py`, and reads/writes data/results from the project folder in Drive. The main thing users need to change is the `PROJECT_PATH` variable in the notebook. 

**Note** If you don't have the writing prompt dataset in your `data` folder, then in the smoke tests, you can just change `--dataset` flag from `writing` to `hc3_all`

## Requirements

- A Google account with access to Google Colab and Google Drive
- Python packages (installed from the notebook):
  - `torch`
  - `numpy`
  - `transformers`
  - `datasets`
  - `matplotlib`
  - `tqdm`
  - `scikit-learn`
  - `openai`
- A Colab GPU runtime (recommended)

## Files You Should Upload to Google Drive

Upload the full `DetectGPT-colab` folder, including:

- `DetectGPT.ipynb`
- `run.py`
- `custom_datasets.py`
- `calculate_metrics.py`
- `requirements.txt`
- `data/` (dataset files)
- `paper_scripts/` (optional helper scripts)

Example Drive layout:

```text
MyDrive/
  DetectGPT-colab/
    DetectGPT.ipynb
    run.py
    custom_datasets.py
    calculate_metrics.py
    requirements.txt
    data/
    paper_scripts/
```

After uploading, remember the full folder path in Drive.

Example:

```python
PROJECT_PATH = "/content/drive/MyDrive/DetectGPT-colab"
```

## Quick Start (Google Colab)

1. Open `DetectGPT.ipynb` from your uploaded Drive folder.
2. In Colab, set runtime to GPU:
   - **Runtime -> Change runtime type -> Hardware accelerator -> GPU**
3. Run the setup cells in order:
   - Install dependencies
   - Import packages
   - Mount Google Drive
   - Set `PROJECT_PATH` to your Drive folder location
4. Run one smoke test cell to verify everything works.

The notebook already includes setup for:

- Installing dependencies
- Mounting Drive
- Setting Hugging Face cache (`HF_HOME`) under your project folder
- Downloading model/dataset assets on first run (then reusing cache)

## Important Configuration

In the setup section of the notebook, update this line to match your Drive location:

```python
PROJECT_PATH = "/content/drive/MyDrive/<YOUR_FOLDER_PATH>/DetectGPT-colab"
```

If this path is wrong, Colab will not find `run.py`, datasets, or output folders.

## Smoke Test Recommendation

Use the provided smoke test cell in the notebook (small sample count) before running medium/full experiments.

A typical smoke test command (already present in the notebook) looks like:

```bash
python run.py --output_name smoke_hc3 --dataset hc3_all --base_model_name gpt2 --mask_filling_model_name t5-small --n_perturbation_list 3 --n_samples 50 --pct_words_masked 0.3 --skip_baselines --cache_dir hf_cache --batch_size 10 --chunk_size 10
```

## Output and Metrics

- Experiment outputs are written under `results/`
- After a run finishes, use `calculate_metrics.py` (also in notebook cells) with the generated JSON result path

## Troubleshooting

- `FileNotFoundError` for scripts/data:
  - Check `PROJECT_PATH` and confirm all files/folders were uploaded to Drive.
- Out-of-memory errors:
  - Use smaller `--batch_size`, `--chunk_size`, or fewer `--n_samples`.
- Slow first run:
  - First execution downloads models/datasets; later runs are faster due to cache.
