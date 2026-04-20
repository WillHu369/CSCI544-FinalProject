#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


THRESHOLDS = {
    "low-fpr": 0.8536432310785527,
    "accuracy": 0.9015310749276843,
}

LABEL_TO_INT = {
    "human": 0,
    "ai": 1,
    "chatgpt": 1,
    "machine": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Binoculars on a CSV and report classification metrics."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input CSV path.")
    parser.add_argument(
        "--text-column",
        default="text",
        help="Name of the column containing text to score.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Name of the column containing human/AI labels.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(THRESHOLDS),
        default="low-fpr",
        help="Built-in Binoculars threshold mode.",
    )
    parser.add_argument(
        "--output-dir",
        default="sample_eval",
        type=Path,
        help="Directory where scored_samples.csv and summary.json are written.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Name for summary.json (defaults to input filename).",
    )
    parser.add_argument(
        "--dataset-used",
        default=None,
        help="Dataset identifier for summary.json (defaults to input file stem).",
    )
    parser.add_argument(
        "--model-used",
        default="",
        help="Model identifier for summary.json (optional).",
    )
    parser.add_argument(
        "--additional-model-used",
        action="append",
        default=[],
        help="Additional model identifier(s) for summary.json (repeatable).",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional notes for summary.json.",
    )
    return parser.parse_args()


def normalize_labels(labels: pd.Series) -> pd.Series:
    cleaned = labels.astype(str).str.strip().str.lower()
    normalized = cleaned.map(LABEL_TO_INT).fillna(cleaned)
    numeric = pd.to_numeric(normalized, errors="raise").astype(int)
    invalid = sorted(set(numeric).difference({0, 1}))
    if invalid:
        raise ValueError(
            "Labels must map to 0/1, human/ai, or human/chatgpt. "
            f"Found invalid values: {invalid}"
        )
    return numeric


def prediction_text(prediction: int) -> str:
    return "Most likely AI-generated" if prediction == 1 else "Most likely human-written"


def predict_from_scores(scores, threshold: float):
    # Binoculars scores below the threshold are classified as AI-generated.
    return (scores < threshold).astype(int)


def compute_auc_roc(y_true, scores) -> Optional[float]:
    # For Binoculars, lower raw scores indicate stronger AI likelihood, so use -score for ROC/AUC.
    try:
        return float(roc_auc_score(y_true, -scores))
    except ValueError:
        return None


def write_roc_curve(output_dir: Path, y_true, scores) -> None:
    fpr, tpr, thresholds = roc_curve(y_true, -scores)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    roc_df.to_csv(output_dir / "roc_curve.csv", sep="\t", index=False)


def select_threshold_max_f1_under_fpr(y_true, scores, max_fpr: float) -> Optional[float]:
    """Select a score threshold that maximizes F1 subject to FPR <= max_fpr."""
    scores_np = np.asarray(scores, dtype=float)
    y_np = np.asarray(y_true, dtype=int)
    if scores_np.size == 0:
        return None

    order = np.argsort(scores_np, kind="mergesort")
    s = scores_np[order]
    y = y_np[order]

    is_pos = y == 1
    is_neg = ~is_pos
    num_pos = int(is_pos.sum())
    num_neg = int(is_neg.sum())

    cum_tp = np.cumsum(is_pos, dtype=np.int64)
    cum_fp = np.cumsum(is_neg, dtype=np.int64)

    unique_scores, first_idx = np.unique(s, return_index=True)
    last_idx = np.r_[first_idx[1:] - 1, s.size - 1]

    tps = cum_tp[last_idx].astype(float)
    fps = cum_fp[last_idx].astype(float)

    # Include the all-negative operating point (threshold below min score).
    tps = np.r_[0.0, tps]
    fps = np.r_[0.0, fps]

    eps = 1e-12
    thresholds = np.r_[float(s[0] - eps), (unique_scores + eps).astype(float)]

    if num_neg > 0:
        fpr = fps / float(num_neg)
    else:
        fpr = np.zeros_like(fps)

    if num_pos > 0:
        recall = tps / float(num_pos)
    else:
        recall = np.zeros_like(tps)

    denom_prec = tps + fps
    precision = np.divide(
        tps,
        denom_prec,
        out=np.zeros_like(tps),
        where=denom_prec > 0,
    )
    denom_f1 = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denom_f1,
        out=np.zeros_like(precision),
        where=denom_f1 > 0,
    )

    valid = fpr <= (max_fpr + 1e-12)
    if not np.any(valid):
        return None

    cand = np.where(valid)[0]
    # Lexicographic max over (f1, recall, -fpr).
    rank = np.lexsort((-fpr[cand], recall[cand], f1[cand]))
    best = int(cand[rank[-1]])
    return float(thresholds[best])


def metrics_at_max_fpr(y_true, scores, max_fpr: float, auc_roc: Optional[float]) -> dict:
    threshold = select_threshold_max_f1_under_fpr(y_true, scores, max_fpr=max_fpr)
    if threshold is None:
        return {
            "f1": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "auc_roc": auc_roc,
        }

    y_pred = predict_from_scores(scores, threshold)
    return {
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auc_roc": auc_roc,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.text_column not in df.columns:
        raise ValueError(f"Missing text column: {args.text_column}")
    if args.label_column not in df.columns:
        raise ValueError(f"Missing label column: {args.label_column}")

    y_true = normalize_labels(df[args.label_column])
    threshold = THRESHOLDS[args.mode]

    from binoculars import Binoculars

    bino = Binoculars(mode=args.mode)
    scores = []
    for text in df[args.text_column].fillna("").astype(str):
        scores.append(float(bino.compute_score(text)))

    scored_df = df.copy()
    scored_df["score"] = scores
    scored_df["prediction"] = predict_from_scores(scored_df["score"], threshold)
    scored_df["prediction_text"] = scored_df["prediction"].map(prediction_text)
    scored_df.to_csv(args.output_dir / "scored_samples.csv", index=False)

    matrix = confusion_matrix(y_true, scored_df["prediction"], labels=[0, 1])
    confusion_df = pd.DataFrame(
        matrix.tolist(),
        index=["actual_human", "actual_ai"],
        columns=["predicted_human", "predicted_ai"],
    )
    confusion_df.to_csv(args.output_dir / "confusion_matrix.csv", index=False)

    auc_roc = compute_auc_roc(y_true, scored_df["score"])
    write_roc_curve(args.output_dir, y_true, scored_df["score"])

    experiment_name = args.experiment_name or args.input.name
    dataset_used = args.dataset_used or args.input.stem

    summary = {
        "experiment_name": experiment_name,
        "detection_method": "Binoculars",
        "model_used": args.model_used or "",
        "dataset_used": dataset_used,
        "num_samples": int(len(scored_df)),
        "additional_details": {
            "additional_models_used": list(args.additional_model_used),
            "notes": args.notes or "",
        },
        "metrics_at_1pct_fpr": metrics_at_max_fpr(
            y_true, scored_df["score"], max_fpr=0.01, auc_roc=auc_roc
        ),
        "metrics_at_0.1pct_fpr": metrics_at_max_fpr(
            y_true, scored_df["score"], max_fpr=0.001, auc_roc=auc_roc
        ),
    }

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

