from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from gpt_zero.config import DEFAULT_TARGET_FPR
from gpt_zero.io_utils import dump_json, ensure_dir, write_table
from gpt_zero.schemas import LABEL_TO_ID


def tpr_at_target_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float = DEFAULT_TARGET_FPR) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid = tpr[fpr <= target_fpr]
    if valid.size == 0:
        return 0.0
    return float(valid.max())


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    target_fpr: float = DEFAULT_TARGET_FPR,
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "tpr_at_1pct_fpr": tpr_at_target_fpr(y_true, y_prob, target_fpr=target_fpr),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "num_samples": int(len(y_true)),
    }


def evaluate_predictions(
    samples: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path | str,
    target_fpr: float = DEFAULT_TARGET_FPR,
) -> dict:
    destination = ensure_dir(output_dir)
    joined = predictions.merge(samples[["sample_id", "label", "domain", "split"]], on="sample_id", how="left")
    joined["y_true"] = joined["label"].map(LABEL_TO_ID)

    summary_rows: list[dict] = []
    domain_rows: list[dict] = []
    roc_rows: list[dict] = []

    for (detector_name, split), group in joined.groupby(["detector_name", "split"], dropna=False):
        y_true = group["y_true"].to_numpy(dtype=int)
        y_prob = group["prob_ai"].to_numpy(dtype=float)
        summary_rows.append(
            {"detector_name": detector_name, "split": split, **compute_binary_metrics(y_true, y_prob, target_fpr=target_fpr)}
        )

        if group["domain"].notna().any():
            for domain, domain_group in group.groupby("domain", dropna=False):
                domain_rows.append(
                    {
                        "detector_name": detector_name,
                        "split": split,
                        "domain": domain,
                        **compute_binary_metrics(
                            domain_group["y_true"].to_numpy(dtype=int),
                            domain_group["prob_ai"].to_numpy(dtype=float),
                            target_fpr=target_fpr,
                        ),
                    }
                )

        if len(np.unique(y_true)) > 1:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            roc_rows.extend(
                {
                    "detector_name": detector_name,
                    "split": split,
                    "fpr": float(fpr_value),
                    "tpr": float(tpr_value),
                    "threshold": float(threshold_value),
                }
                for fpr_value, tpr_value, threshold_value in zip(fpr, tpr, thresholds)
            )

    summary_frame = pd.DataFrame(summary_rows)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values(["split", "detector_name"]).reset_index(drop=True)

    domain_frame = pd.DataFrame(domain_rows)
    if not domain_frame.empty:
        domain_frame = domain_frame.sort_values(["split", "detector_name", "domain"]).reset_index(drop=True)

    roc_frame = pd.DataFrame(roc_rows)
    if not roc_frame.empty:
        roc_frame = roc_frame.sort_values(["split", "detector_name", "fpr"]).reset_index(drop=True)

    write_table(summary_frame, Path(destination) / "metrics_summary.csv")
    write_table(domain_frame, Path(destination) / "metrics_by_domain.csv")
    write_table(roc_frame, Path(destination) / "roc_points.csv")

    summary = {
        "num_predictions": int(len(predictions)),
        "num_detectors": int(predictions["detector_name"].nunique()),
        "splits": sorted(samples["split"].dropna().astype(str).unique().tolist()),
        "target_fpr": target_fpr,
    }
    dump_json(summary, Path(destination) / "summary.json")
    return summary
