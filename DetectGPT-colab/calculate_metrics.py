import json
import argparse
import csv
import re
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
    args_path = results_path.parent / "args.json"
    args_dict = {}
    if args_path.exists():
        with open(args_path, "r", encoding="utf-8") as f:
            args_dict = json.load(f)

    mask_model_name = args_dict.get("mask_filling_model_name")
    additional_models = []
    if isinstance(mask_model_name, str) and mask_model_name.strip():
        additional_models = [mask_model_name]

    return {
        "detection_method": "DetectGPT",
        "model_used": args_dict.get("base_model_name", results_dict.get("model_used", "unknown")),
        "dataset_used": args_dict.get("dataset", results_dict.get("dataset_used", "unknown")),
        "additional_models_used": additional_models or results_dict.get("additional_models_used", []),
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


def compute_binary_metrics_at_target_fpr(y_true, y_score, target_fpr):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = select_threshold_at_target_fpr(fpr, tpr, thresholds, target_fpr=target_fpr)

    thr = thresholds[idx]
    y_pred = (y_score >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return {
        "target_fpr": float(target_fpr),
        "selected_threshold": float(thr),
        "actual_fpr": float(fpr[idx]),
        "actual_tpr": float(tpr[idx]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def write_roc_curve_csv(fpr, tpr, thresholds, out_path):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fpr", "tpr", "threshold"])
        for fpr_i, tpr_i, thr_i in zip(fpr, tpr, thresholds):
            writer.writerow([float(fpr_i), float(tpr_i), float(thr_i)])


def sanitize_filename_part(value):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return cleaned.strip("._-") or "unknown"


def infer_timestamp(results_file_path):
    # DetectGPT runs store outputs under a timestamped directory; use that as timestamp.
    parent_name = results_file_path.parent.name
    if parent_name:
        return parent_name
    return "unknown"

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
fpr_curve, tpr_curve, thresholds_curve = roc_curve(y_true, y_score)
roc_auc = roc_auc_score(y_true, y_score)

# Evaluate at 1% FPR (0.01) and 0.1% FPR (0.001).
metrics_at_1pct = compute_binary_metrics_at_target_fpr(y_true, y_score, target_fpr=0.01)
metrics_at_0_1pct = compute_binary_metrics_at_target_fpr(y_true, y_score, target_fpr=0.001)

print(
    "Selected threshold for target_fpr(1.00%): "
    f"{metrics_at_1pct['selected_threshold']:.4f} "
    f"(FPR={metrics_at_1pct['actual_fpr']:.4f}, TPR={metrics_at_1pct['actual_tpr']:.4f})"
)
print(
    "Selected threshold for target_fpr(0.10%): "
    f"{metrics_at_0_1pct['selected_threshold']:.4f} "
    f"(FPR={metrics_at_0_1pct['actual_fpr']:.4f}, TPR={metrics_at_0_1pct['actual_tpr']:.4f})"
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
    "metrics_at_1pct_fpr": {
        "f1": metrics_at_1pct["f1"],
        "accuracy": metrics_at_1pct["accuracy"],
        "precision": metrics_at_1pct["precision"],
        "recall": metrics_at_1pct["recall"],
        "tpr": metrics_at_1pct["actual_tpr"],
        "auc_roc": float(roc_auc)
    },
    "metrics_at_0.1pct_fpr": {
        "f1": metrics_at_0_1pct["f1"],
        "accuracy": metrics_at_0_1pct["accuracy"],
        "precision": metrics_at_0_1pct["precision"],
        "recall": metrics_at_0_1pct["recall"],
        "tpr": metrics_at_0_1pct["actual_tpr"],
        "auc_roc": float(roc_auc)
    }
}

output_dir = Path(__file__).resolve().parent / "metric_results"
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = sanitize_filename_part(infer_timestamp(results_path))
model_name = sanitize_filename_part(metadata["model_used"])
dataset_name = sanitize_filename_part(metadata["dataset_used"])
output_path = output_dir / f"{model_name}_{dataset_name}_{timestamp}_results.json"
roc_csv_filename = (
    f"{sanitize_filename_part(metadata['detection_method'])}_"
    f"{sanitize_filename_part(metadata['model_used'])}_"
    f"{sanitize_filename_part(metadata['dataset_used'])}.csv"
)
roc_csv_path = output_dir / roc_csv_filename

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_payload, f, indent=2)

write_roc_curve_csv(fpr_curve, tpr_curve, thresholds_curve, roc_csv_path)

print(json.dumps(output_payload, indent=2))