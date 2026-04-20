#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ROC curves for multiple detectors on the baseline (original clean) dataset."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root() / "Metrics" / "plots" / "roc_curves_original_clean.png",
        help="Output PNG path (default: Metrics/plots/roc_curves_original_clean.png).",
    )
    parser.add_argument(
        "--baseline-name",
        default="hc3_unified_10000_seed42_clean_test",
        help="dataset_name value to filter the ZeroGPT ROC points CSV (default: hc3_unified_10000_seed42_clean_test).",
    )
    return parser.parse_args()


def read_roc_csv(path: Path, sep: str = ",") -> Tuple["pd.Series", "pd.Series"]:
    import pandas as pd

    df = pd.read_csv(path, sep=sep)
    if not {"fpr", "tpr"}.issubset(df.columns):
        raise ValueError(f"ROC CSV missing fpr/tpr columns: {path}")
    return df["fpr"], df["tpr"]


def load_binoculars(metrics_dir: Path) -> Optional[Tuple["pd.Series", "pd.Series"]]:
    path = metrics_dir / "Binoculars" / "Binoculars-10000.csv"
    if not path.exists():
        return None
    return read_roc_csv(path, sep="\t")


def load_detectgpt(metrics_dir: Path) -> Dict[str, Tuple["pd.Series", "pd.Series"]]:
    out: Dict[str, Tuple["pd.Series", "pd.Series"]] = {}
    roc_dir = metrics_dir / "DetectGPT" / "ROC_curve_csv"
    if not roc_dir.exists():
        return out

    mapping = {
        "DetectGPT (GPT2-Large)": "DetectGPT_openai-community_gpt2-large_hc3_all.csv",
        "DetectGPT (Falcon-7B)": "DetectGPT_tiiuae_falcon-7b_hc3_all.csv",
        "DetectGPT (Falcon-Instruct)": "DetectGPT_tiiuae_falcon-7b-instruct_hc3_all.csv",
    }
    for label, filename in mapping.items():
        path = roc_dir / filename
        if not path.exists():
            continue
        out[label] = read_roc_csv(path, sep=",")
    return out


def load_zerogpt_points(
    metrics_dir: Path, baseline_name: str
) -> Dict[str, Tuple["pd.Series", "pd.Series"]]:
    import pandas as pd

    path = metrics_dir / "ZeroGPT" / "metrics_share" / "all_test_dataset_roc_points.csv"
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    required = {"detector_name", "split", "fpr", "tpr", "dataset_name"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"ZeroGPT ROC points missing columns {missing}: {path}")

    df = df[(df["split"] == "test") & (df["dataset_name"] == baseline_name)]
    if df.empty:
        return {}

    det_label = {
        "gptzero_like": "GPTZero",
        "xgboost_tfidf": "XGB",
        "svm_tfidf": "SVM",
    }

    out: Dict[str, Tuple["pd.Series", "pd.Series"]] = {}
    for det_name, group in df.groupby("detector_name", sort=True):
        label = det_label.get(det_name, det_name)
        out[label] = (group["fpr"], group["tpr"])
    return out


def main() -> None:
    args = parse_args()
    metrics_dir = repo_root() / "Metrics"

    # Set before importing matplotlib to avoid ~/.matplotlib permission warnings.
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves: Dict[str, Tuple["pd.Series", "pd.Series"]] = {}

    bino = load_binoculars(metrics_dir)
    if bino is not None:
        curves["Binoculars"] = bino

    curves.update(load_detectgpt(metrics_dir))
    curves.update(load_zerogpt_points(metrics_dir, baseline_name=args.baseline_name))

    if not curves:
        raise SystemExit("No ROC curve sources found under Metrics/.")

    # Color map consistent with our other plots.
    method_order = [
        "DetectGPT (GPT2-Large)",
        "DetectGPT (Falcon-7B)",
        "DetectGPT (Falcon-Instruct)",
        "Binoculars",
        "GPTZero",
        "XGB",
        "SVM",
    ]
    cmap = plt.get_cmap("tab10")
    color_by_label = {label: cmap(i % 10) for i, label in enumerate(method_order)}

    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=180)

    # Chance line.
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="0.6", label="Chance")

    for label in method_order:
        if label not in curves:
            continue
        fpr, tpr = curves[label]
        ax.plot(
            fpr,
            tpr,
            linewidth=1.6,
            color=color_by_label.get(label),
            label=label,
        )

    # Add anything else we found but didn't know about.
    for label, (fpr, tpr) in curves.items():
        if label in method_order:
            continue
        ax.plot(fpr, tpr, linewidth=1.4, label=label)

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curves (Baseline 10000)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="lower right", fontsize=8, frameon=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

