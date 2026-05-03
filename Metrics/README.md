# Metrics

This folder contains the detector outputs (JSON summaries, ROC points, and plots) used for our writeup and figures.

## Layout

Key subfolders:

- `Metrics/Binoculars/`: Binoculars summaries and ROC-point CSVs
- `Metrics/DetectGPT/`: DetectGPT summaries and ROC curve CSVs
- `Metrics/metrics_XGB_SVM_GPTZERO/`: classical baselines + GPTZero-like JSON summaries
- `Metrics/plots/`: generated tables and figure PNGs

## Metric format

Most detector runs export JSON with low-FPR operating points, for example:

- `metrics_at_0.1pct_fpr` (FPR <= 0.001)
- `metrics_at_1pct_fpr` (FPR <= 0.01)

For detectors where recall corresponds to TPR, we also store `tpr` alongside `recall`.

ROC points are stored as CSV with `fpr` and `tpr` (and usually `threshold`).

## How to generate plots

Run from the repo root.

### Bar plots + tables (TPR/F1/AUC)

```bash
python Metrics/make_metric_graphs.py --variants original_clean
python Metrics/make_metric_graphs.py --variants stylistic_cleanup
python Metrics/make_metric_graphs.py --variants paraphrasing
```

Outputs are written under:

```text
Metrics/plots/
```

### ROC curve comparisons

Plot multiple detectors' ROC curves on one figure:

```bash
python Metrics/plot_roc_comparison.py --variant original_clean
python Metrics/plot_roc_comparison.py --variant stylistic_cleanup --detectors binoculars,gptzero,xgb,svm
python Metrics/plot_roc_comparison.py --variant paraphrasing --detectors binoculars,gptzero,xgb,svm
```

## Notes

- The scripts in this folder read the tracked JSON/CSV metric exports and only generate figures and tables. They do not run detectors.
