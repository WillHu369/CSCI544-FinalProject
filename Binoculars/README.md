# Binoculars

This folder contains our Binoculars evaluation pipeline. Binoculars is a zero-shot detector that scores text by contrasting two closely related language models (an observer and a performer) using a perplexity vs. cross-perplexity signal.

## Environment setup

We ran Binoculars in Google Colab using a GPU runtime.

The tracked notebook in this repo is:

```text
Binoculars/Binoculars_Colab.ipynb
```

## Device / system used

- Platform: Google Colab
- Runtime: GPU runtime (A100 recommended for Falcon-7B weights)

## Input files

Binoculars can be evaluated on either:

1. an HC3 "wide" CSV (recommended) with columns:
   `hc3_row_id`, `source`, `question`, `human_answers`, `chatgpt_answers`
2. a flat CSV with `text` and `label` (where label maps to human/ai or 0/1)

The canonical HC3 inputs in this repo live under:

```text
HC3-Dataset-Samples/
```

## How to run

1. Open `Binoculars/Binoculars_Colab.ipynb` in Colab.
2. Select a GPU runtime.
3. Upload:
   - one or more HC3 CSVs from `HC3-Dataset-Samples/` (or any `text`/`label` CSV), and
   - `Binoculars/evaluate_samples.py`
4. Run the notebook cells in order.

## Outputs

For each uploaded CSV, the notebook runs `evaluate_samples.py` and writes an eval folder:

- `scored_samples.csv` (per-sample scores + predictions)
- `confusion_matrix.csv`
- `roc_curve.csv` (tab-separated: `fpr`, `tpr`, `threshold`)
- `summary.json` (aggregated metrics at low-FPR operating points)

The notebook also downloads a zip of the per-run folder so `roc_curve.csv` is not missed.

## Tracked results

The results used for figures and reporting are kept under:

```text
Metrics/Binoculars/
```

These include JSON summaries (F1/TPR/AUC at constrained FPR) and ROC-point CSVs for:

- baseline/original clean
- stylistic cleanup
- T5 perturbations
- recursive paraphrasing
