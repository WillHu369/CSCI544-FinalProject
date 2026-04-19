from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from gpt_zero.config import DEFAULT_TARGET_FPR
from gpt_zero.io_utils import dump_json, ensure_dir, write_table
from gpt_zero.schemas import LABEL_TO_ID


def target_fpr_metric_name(target_fpr: float) -> str:
    return fixed_fpr_metric_name("tpr", target_fpr)


def fixed_fpr_metric_name(metric_name: str, target_fpr: float) -> str:
    percent = float(target_fpr) * 100
    percent_text = f"{percent:.8g}".replace("-", "neg_").replace(".", "_")
    return f"{metric_name}_at_{percent_text}pct_fpr"


def _normalize_target_fprs(
    target_fpr: float | None,
    target_fprs: Iterable[float] | None = None,
) -> tuple[float, ...]:
    values: list[float] = []
    if target_fpr is not None:
        values.append(float(target_fpr))
    if target_fprs is not None:
        values.extend(float(value) for value in target_fprs)
    if not values:
        values.append(float(DEFAULT_TARGET_FPR))

    normalized: list[float] = []
    for value in values:
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"target FPR values must be finite values in [0, 1], got {value!r}")
        if not any(np.isclose(value, existing) for existing in normalized):
            normalized.append(value)
    return tuple(normalized)


def tpr_at_target_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float = DEFAULT_TARGET_FPR) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid = tpr[fpr <= target_fpr]
    if valid.size == 0:
        return 0.0
    return float(valid.max())


def _operating_threshold_at_target_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr: float,
) -> tuple[float, float, float]:
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid_indices = np.flatnonzero(fpr <= target_fpr)
    if valid_indices.size == 0:
        return float("nan"), float("nan"), float("nan")

    best_tpr = tpr[valid_indices].max()
    best_indices = valid_indices[np.isclose(tpr[valid_indices], best_tpr)]
    selected_index = best_indices[np.argmax(fpr[best_indices])]
    return float(thresholds[selected_index]), float(fpr[selected_index]), float(tpr[selected_index])


def _classification_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def fixed_fpr_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr: float,
) -> dict:
    suffix_metrics = {
        fixed_fpr_metric_name("threshold", target_fpr): float("nan"),
        fixed_fpr_metric_name("actual_fpr", target_fpr): float("nan"),
        target_fpr_metric_name(target_fpr): float("nan"),
        fixed_fpr_metric_name("accuracy", target_fpr): float("nan"),
        fixed_fpr_metric_name("precision", target_fpr): float("nan"),
        fixed_fpr_metric_name("recall", target_fpr): float("nan"),
        fixed_fpr_metric_name("f1", target_fpr): float("nan"),
    }
    threshold, actual_fpr, actual_tpr = _operating_threshold_at_target_fpr(y_true, y_prob, target_fpr)
    if np.isnan(threshold):
        return suffix_metrics

    threshold_metrics = _classification_metrics_at_threshold(y_true, y_prob, threshold)
    suffix_metrics.update(
        {
            fixed_fpr_metric_name("threshold", target_fpr): threshold,
            fixed_fpr_metric_name("actual_fpr", target_fpr): actual_fpr,
            target_fpr_metric_name(target_fpr): actual_tpr,
            fixed_fpr_metric_name("accuracy", target_fpr): threshold_metrics["accuracy"],
            fixed_fpr_metric_name("precision", target_fpr): threshold_metrics["precision"],
            fixed_fpr_metric_name("recall", target_fpr): threshold_metrics["recall"],
            fixed_fpr_metric_name("f1", target_fpr): threshold_metrics["f1"],
        }
    )
    return suffix_metrics


def _resolve_prediction_threshold(group: pd.DataFrame) -> float:
    if "decision_threshold" not in group.columns:
        return 0.5
    thresholds = pd.to_numeric(group["decision_threshold"], errors="coerce").dropna()
    if thresholds.empty:
        return 0.5
    return float(thresholds.median())


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    target_fpr: float = DEFAULT_TARGET_FPR,
    target_fprs: Iterable[float] | None = None,
) -> dict:
    threshold_metrics = _classification_metrics_at_threshold(y_true, y_prob, threshold)
    metrics = {
        "accuracy": threshold_metrics["accuracy"],
        "precision": threshold_metrics["precision"],
        "recall": threshold_metrics["recall"],
        "f1": threshold_metrics["f1"],
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    for fixed_fpr in _normalize_target_fprs(target_fpr, target_fprs):
        metrics.update(fixed_fpr_classification_metrics(y_true, y_prob, fixed_fpr))
    metrics.update(
        {
            "threshold": float(threshold),
            "tn": threshold_metrics["tn"],
            "fp": threshold_metrics["fp"],
            "fn": threshold_metrics["fn"],
            "tp": threshold_metrics["tp"],
            "num_samples": int(len(y_true)),
        }
    )
    return metrics


def evaluate_predictions(
    samples: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path | str,
    target_fpr: float = DEFAULT_TARGET_FPR,
    target_fprs: Iterable[float] | None = None,
) -> dict:
    destination = ensure_dir(output_dir)
    joined = predictions.merge(samples[["sample_id", "label", "domain", "split"]], on="sample_id", how="left")
    joined["y_true"] = joined["label"].map(LABEL_TO_ID)
    fixed_fprs = _normalize_target_fprs(target_fpr, target_fprs)

    summary_rows: list[dict] = []
    domain_rows: list[dict] = []
    roc_rows: list[dict] = []

    for (detector_name, split), group in joined.groupby(["detector_name", "split"], dropna=False):
        y_true = group["y_true"].to_numpy(dtype=int)
        y_prob = group["prob_ai"].to_numpy(dtype=float)
        threshold = _resolve_prediction_threshold(group)
        summary_rows.append(
            {
                "detector_name": detector_name,
                "split": split,
                **compute_binary_metrics(y_true, y_prob, threshold=threshold, target_fpr=None, target_fprs=fixed_fprs),
            }
        )

        if group["domain"].notna().any():
            for domain, domain_group in group.groupby("domain", dropna=False):
                domain_threshold = _resolve_prediction_threshold(domain_group)
                domain_rows.append(
                    {
                        "detector_name": detector_name,
                        "split": split,
                        "domain": domain,
                        **compute_binary_metrics(
                            domain_group["y_true"].to_numpy(dtype=int),
                            domain_group["prob_ai"].to_numpy(dtype=float),
                            threshold=domain_threshold,
                            target_fpr=None,
                            target_fprs=fixed_fprs,
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
        "target_fprs": list(fixed_fprs),
    }
    dump_json(summary, Path(destination) / "summary.json")
    return summary
