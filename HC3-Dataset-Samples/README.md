# HC3 Dataset Samples — Notes

This folder is for the subset of HC3 examples used in our experiments (human vs model-generated passages). The goal is to keep the exact evaluation inputs stable so detector results are comparable across runs.

## Environment setup

No special environment is required to *store* or *read* the samples in this folder. To generate a fresh sample from HC3, you need Python plus the Hugging Face `datasets` library.

```bash
pip install datasets tqdm
```

## Device / system used

I generated the samples locally on my laptop (CPU-only) and waited for the download + sampling to finish.

If you want this to be fully reproducible, fill in:
- OS:
- CPU:
- RAM:
- Python version:

## Instructions for running the code

### Generate 10,000 HC3 rows

Run from the repo root:

```bash
python HC3-dataset-samples/generate_hc3_samples.py \
  --total-samples 10000 \
  --out HC3-dataset-samples/hc3_10000.jsonl
```

### Generate a CSV (recommended for uploads)

```bash
python HC3-dataset-samples/generate_hc3_samples.py \
  --total-samples 10000 \
  --format csv \
  --out HC3-dataset-samples/hc3_10000.csv
```

Notes:
- This script downloads HC3 via Hugging Face, so it requires internet access.
- By default it tries to balance `human` vs `ai` rows (one answer per row). Use `--no-balanced` to disable.

## How results are generated

The “result” of running the sampling script is the exported file (`.jsonl` or `.csv`). Each output row corresponds to a single (question, answer) pair and includes a `label` field (`human` or `ai`), plus metadata like the dataset config and example index. These exported rows are then used as the fixed evaluation inputs for the detectors (for example, Binoculars; see `binoculars/README.md`).
