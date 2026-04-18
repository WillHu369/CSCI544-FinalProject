#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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


def f1_at_max_fpr(y_true, scores, max_fpr: float = 0.0001) -> dict:
    """Compute the best F1 among thresholds whose human false-positive rate is <= max_fpr."""
    candidates = sorted(set(float(score) for score in scores))
    if not candidates:
        return {
            "f1": None,
            "threshold": None,
            "max_fpr": max_fpr,
            "actual_fpr": None,
            "note": "No scores were available.",
        }

    # Add edge thresholds so all-human and all-AI predictions are considered.
    epsilon = 1e-12
    thresholds = [min(candidates) - epsilon]
    thresholds.extend(score + epsilon for score in candidates)

    best = None
    for threshold in thresholds:
        y_pred = predict_from_scores(scores, threshold)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        actual_fpr = fp / (fp + tn) if (fp + tn) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        if actual_fpr > max_fpr:
            continue

        if best is None or (f1, recall, -actual_fpr) > (
            best["f1"],
            best["recall"],
            -best["actual_fpr"],
        ):
            best = {
                "f1": float(f1),
                "threshold": float(threshold),
                "max_fpr": float(max_fpr),
                "actual_fpr": float(actual_fpr),
                "recall": float(recall),
            }

    if best is None:
        return {
            "f1": None,
            "threshold": None,
            "max_fpr": float(max_fpr),
            "actual_fpr": None,
            "recall": None,
            "note": "No threshold satisfied the target FPR.",
        }

    return best


def compute_metrics(y_true, scores, threshold: float) -> dict:
    y_pred = predict_from_scores(scores, threshold)
    labels = [0, 1]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    try:
        roc_auc = float(roc_auc_score(y_true, -scores))
    except ValueError:
        roc_auc = None

    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_at_1pct_fpr": f1_at_max_fpr(y_true, scores, max_fpr=0.01),
        "f1_at_0_01pct_fpr": f1_at_max_fpr(y_true, scores, max_fpr=0.0001),
        "auc_roc": roc_auc,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": ["human", "ai"],
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
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

    metrics = compute_metrics(y_true, scored_df["score"], threshold)
    confusion_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["actual_human", "actual_ai"],
        columns=["predicted_human", "predicted_ai"],
    )
    confusion_df.to_csv(args.output_dir / "confusion_matrix.csv")

    summary = {
        "count": int(len(scored_df)),
        "mode": args.mode,
        "threshold": float(threshold),
        **metrics,
        # Report-friendly metric labels.
        "F1@1%FPR": metrics["f1_at_1pct_fpr"]["f1"],
        "F1@0.01%FPR": metrics["f1_at_0_01pct_fpr"]["f1"],
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "AUC-ROC": metrics["auc_roc"],
        "Confusion Matrix": metrics["confusion_matrix"],
        # Backwards-compatible aliases used by earlier notebook versions.
        "precision_ai": metrics["precision"],
        "recall_ai": metrics["recall"],
        "f1_ai": metrics["f1"],
        "roc_auc": metrics["auc_roc"],
    }

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
