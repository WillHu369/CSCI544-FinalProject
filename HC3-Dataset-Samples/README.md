# HC3 Dataset Samples

This folder contains the HC3 subsets used across our experiments. The goal is to keep evaluation inputs stable so detector results are comparable across runs and avoidance techniques.

## Environment setup

To regenerate samples from HC3 you need Python plus:

```bash
pip install pandas huggingface_hub
```

The script downloads HC3 via Hugging Face, so it requires internet access.

## Device / system used

- Platform used for the tracked exports: local CPU-only environment

## Input files

The sampling script pulls the full dataset from:

```text
Hello-SimpleAI/HC3 (all.jsonl)
```

## How to run

All commands below assume you run from the repo root.

### 1) Create a unified sample with N total rows

Create exactly 10,000 HC3 rows:

```bash
python HC3-Dataset-Samples/create_hc3_sample.py --total-samples 10000 --seed 42 --output-dir HC3-Dataset-Samples
```

To expand a previous unified set while preserving the original rows (e.g., keep the existing 1,000 rows inside a new 10,000-row sample):

```bash
python HC3-Dataset-Samples/create_hc3_sample.py \
  --total-samples 10000 \
  --seed 42 \
  --preserve-unified-sample HC3-Dataset-Samples/hc3_unified_1000_seed42.csv \
  --output-dir HC3-Dataset-Samples
```

### 2) Create a balanced per-domain sample

Sample the same number of rows from each HC3 `source`:

```bash
python HC3-Dataset-Samples/create_hc3_sample.py --samples-per-domain 200 --seed 42 --output-dir HC3-Dataset-Samples
```

## Outputs

The script always writes:

1. one unified CSV, named like:
   `hc3_unified_<N>_seed<seed>.csv`
2. one CSV per HC3 source, named like:
   `hc3_<source>_<count>_seed<seed>.csv`

## File format

Each output row is an HC3 "wide" row with these columns:

- `hc3_row_id`
- `source`
- `question`
- `human_answers` (JSON-serialized list)
- `chatgpt_answers` (JSON-serialized list)

Downstream evaluation scripts (e.g., Binoculars in `Binoculars/evaluate_samples.py`) expand the answer-list columns into per-answer `text` + `label` rows for scoring.
