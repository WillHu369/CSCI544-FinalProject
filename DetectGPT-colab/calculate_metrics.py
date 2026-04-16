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


def infer_metadata(results_dict):
    return {
        "detection_method": results_dict.get("detection_method", "unknown"),
        "model_used": results_dict.get("model_used", "unknown"),
        "dataset_used": results_dict.get("dataset_used", "unknown"),
        "additional_models_used": results_dict.get("additional_models_used", []),
        "notes": results_dict.get("notes", "")
    }


def select_threshold_at_target_fpr(fpr, tpr, thresholds, target_fpr=0.01):
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        # If no ROC point is at or below target FPR, pick the closest point.
        idx = int(np.argmin(np.abs(fpr - target_fpr)))
    else:
        idx = int(valid[np.argmax(tpr[valid])])
    return idx

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

# Pick threshold at 1% FPR operating region.
target_fpr = 0.01
idx = select_threshold_at_target_fpr(fpr, tpr, thresholds, target_fpr=target_fpr)

thr = thresholds[idx]
print(f"Selected threshold for target_fpr({target_fpr:.2f}): {thr:.4f} (FPR={fpr[idx]:.4f}, TPR={tpr[idx]:.4f})")
y_pred = (y_score >= thr).astype(int)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary", zero_division=0
)

metadata = infer_metadata(d)
output_payload = {
    "experiment_name": results_path.name,
    "detection_method": metadata["detection_method"],
    "model_used": metadata["model_used"],
    "dataset_used": metadata["dataset_used"],
    "num_samples": int(len(y_true)),
    "additional_details": {
        "additional_models_used": metadata["additional_models_used"],
        "notes": metadata["notes"]
    },
    "metrics": {
        "f1_at_1pct_fpr": float(f1),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "auc_roc": float(roc_auc)
    }
}

output_dir = Path(__file__).resolve().parent / "metric_results"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"RESULTS_{results_path.stem}.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_payload, f, indent=2)

print("ROC-AUC:", roc_auc)
print("threshold:", thr)
print("confusion matrix [[TN, FP], [FN, TP]] =", [[tn, fp], [fn, tp]])
print("accuracy:", acc, "precision:", prec, "recall:", rec, "f1:", f1)
print("Saved metrics JSON to:", output_path)