import json
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)

parser = argparse.ArgumentParser(description="Compute metrics from a perturbation results JSON file")
parser.add_argument(
    "--path",
    type=str,
    required=True,
    help="Path to perturbation_*_results.json"
)
args = parser.parse_args()

def resolve_results_path(path_str):
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p

    cwd = Path.cwd()
    candidates = [cwd / p, cwd.parent / p]

    if p.parts and p.parts[0].lower() == "detect-gpt":
        trimmed = Path(*p.parts[1:]) if len(p.parts) > 1 else Path()
        candidates.extend([cwd / trimmed, cwd.parent / trimmed])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    attempted = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find file from --path={path_str}. Tried:\n{attempted}"
    )


results_path = resolve_results_path(args.path)

with open(results_path, "r", encoding="utf-8") as f:
    d = json.load(f)

real = np.array(d["predictions"]["real"], dtype=float)      # label 0
samples = np.array(d["predictions"]["samples"], dtype=float) # label 1

y_true = np.array([0]*len(real) + [1]*len(samples))
y_score = np.concatenate([real, samples])

# Threshold-free metrics
roc_auc = roc_auc_score(y_true, y_score)
fpr, tpr, thresholds = roc_curve(y_true, y_score)

# TPR@1%FPR (piecewise ROC points)
mask = fpr <= 0.01
tpr_at_1pct = float(np.max(tpr[mask])) if np.any(mask) else 0.0

# Pick a threshold. Threshold at 5% FPR:
target_fpr = 0.1
valid = np.where(fpr <= target_fpr)[0]

if len(valid) == 0:
    idx = 0
else:
    idx = valid[np.argmax(tpr[valid])]

thr = thresholds[idx]
print(f"Selected threshold for target_fpr({target_fpr:.2f}): {thr:.4f} (FPR={fpr[idx]:.4f}, TPR={tpr[idx]:.4f})")
y_pred = (y_score >= thr).astype(int)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary", zero_division=0
)

print("ROC-AUC:", roc_auc)
print("TPR@1%FPR:", tpr_at_1pct)
print("threshold:", thr)
print("confusion matrix [[TN, FP], [FN, TP]] =", [[tn, fp], [fn, tp]])
print("accuracy:", acc, "precision:", prec, "recall:", rec, "f1:", f1)