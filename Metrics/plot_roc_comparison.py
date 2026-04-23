#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ROC curves for multiple detectors on the requested dataset variant."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (defaults to Metrics/plots/roc_curves_<variant>.png).",
    )
    parser.add_argument(
        "--variant",
        choices=["original_clean", "stylistic_cleanup", "paraphrasing"],
        default="original_clean",
        help="Dataset variant to plot (default: original_clean).",
    )
    parser.add_argument(
        "--detectors",
        default="detectgpt_gpt2_large,detectgpt_falcon_7b,detectgpt_falcon_instruct,binoculars,gptzero,xgb,svm",
        help="Comma-separated detectors to include (default: include all available).",
    )
    return parser.parse_args()


def read_roc_csv(path: Path, sep: str = ",") -> Tuple["pd.Series", "pd.Series"]:
    import pandas as pd

    df = pd.read_csv(path, sep=sep)
    if not {"fpr", "tpr"}.issubset(df.columns):
        raise ValueError(f"ROC CSV missing fpr/tpr columns: {path}")
    return df["fpr"], df["tpr"]


def load_binoculars(metrics_dir: Path, variant: str) -> Optional[Tuple["pd.Series", "pd.Series"]]:
    if variant == "stylistic_cleanup":
        path = metrics_dir / "Binoculars" / "Binoculars-stylistic-cleanup.csv"
    elif variant == "paraphrasing":
        path = metrics_dir / "Binoculars" / "Binoculars-paraphrasing.csv"
    else:
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
        "detectgpt_gpt2_large": (
            "DetectGPT (GPT2-Large)",
            "DetectGPT_openai-community_gpt2-large_hc3_all.csv",
        ),
        "detectgpt_falcon_7b": (
            "DetectGPT (Falcon-7B)",
            "DetectGPT_tiiuae_falcon-7b_hc3_all.csv",
        ),
        "detectgpt_falcon_instruct": (
            "DetectGPT (Falcon-Instruct)",
            "DetectGPT_tiiuae_falcon-7b-instruct_hc3_all.csv",
        ),
    }
    for _, (pretty, file) in mapping.items():
        path = roc_dir / file
        if not path.exists():
            continue
        out[pretty] = read_roc_csv(path, sep=",")
    return out


def load_zerogpt_points(
    metrics_dir: Path, dataset_name: str
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

    df = df[(df["split"] == "test") & (df["dataset_name"] == dataset_name)]
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

    variant_to_dataset_name = {
        "original_clean": "hc3_unified_10000_seed42_clean_test",
        "stylistic_cleanup": "hc3_stylistic_cleanup_10000_clean_test",
        "paraphrasing": "hc3_gpt_5_4_mini_recursive_paraphrase_depth_3_10000_clean_test",
    }
    dataset_name = variant_to_dataset_name[args.variant]

    requested = [d.strip() for d in args.detectors.split(",") if d.strip()]
    want = set(requested)

    curves: Dict[str, Tuple["pd.Series", "pd.Series"]] = {}

    if "binoculars" in want:
        bino = load_binoculars(metrics_dir, variant=args.variant)
        if bino is not None:
            curves["Binoculars"] = bino

    # DetectGPT ROC curves are only available for the original_clean dataset.
    if args.variant == "original_clean" and any(d.startswith("detectgpt_") for d in want):
        detect_curves = load_detectgpt(metrics_dir)
        key_to_label = {
            "detectgpt_gpt2_large": "DetectGPT (GPT2-Large)",
            "detectgpt_falcon_7b": "DetectGPT (Falcon-7B)",
            "detectgpt_falcon_instruct": "DetectGPT (Falcon-Instruct)",
        }
        for key, pretty in key_to_label.items():
            if key in want and pretty in detect_curves:
                curves[pretty] = detect_curves[pretty]

    zero_curves = load_zerogpt_points(metrics_dir, dataset_name=dataset_name)
    if "gptzero" in want and "GPTZero" in zero_curves:
        curves["GPTZero"] = zero_curves["GPTZero"]
    if "xgb" in want and "XGB" in zero_curves:
        curves["XGB"] = zero_curves["XGB"]
    if "svm" in want and "SVM" in zero_curves:
        curves["SVM"] = zero_curves["SVM"]

    if not curves:
        raise SystemExit("No ROC curve sources found under Metrics/.")

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

    for label, (fpr, tpr) in curves.items():
        if label in method_order:
            continue
        ax.plot(fpr, tpr, linewidth=1.4, label=label)

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    if args.variant == "original_clean":
        title = "ROC Curves (Baseline 10000)"
    elif args.variant == "stylistic_cleanup":
        title = "ROC Curves (Stylistic Cleanup 10000)"
    else:
        title = "ROC Curves (Paraphrasing 10000)"
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="lower right", fontsize=8, frameon=True)

    output = args.output
    if output is None:
        output = repo_root() / "Metrics" / "plots" / f"roc_curves_{args.variant}.png"

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

