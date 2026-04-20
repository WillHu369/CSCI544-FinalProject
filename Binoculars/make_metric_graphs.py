#!/usr/bin/env python3

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str


METHODS: List[MethodSpec] = [
    MethodSpec("detectgpt_gpt2_large", "DetectGPT (GPT2-Large)"),
    MethodSpec("detectgpt_falcon_7b", "DetectGPT (Falcon-7B)"),
    MethodSpec("detectgpt_falcon_instruct", "DetectGPT (Falcon-Instruct)"),
    MethodSpec("binoculars", "Binoculars"),
    MethodSpec("gptzero", "GPTZero"),
    MethodSpec("xgb", "XGB"),
    MethodSpec("svm", "SVM"),
]

VARIANT_LABELS = {
    "original_clean": "Baseline 10000",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create TPR@0.1%FPR and F1@0.1%FPR comparison plots from Metrics/*.json files."
    )
    parser.add_argument(
        "--variants",
        default="original_clean",
        help="Comma-separated variants to plot (e.g. original_clean,stylistic_cleanup). Default: original_clean",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=repo_root() / "Metrics",
        help="Metrics directory (default: ./Metrics).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "Metrics" / "plots",
        help="Directory to write plots and tables (default: ./Metrics/plots).",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def metric_get(d: Dict[str, Any], *keys: str) -> Optional[float]:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if cur is None:
        return None
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def method_paths(metrics_dir: Path, variant: str) -> Dict[str, Path]:
    # Keep this mapping explicit so the plot matches the report.
    paths: Dict[str, Path] = {}

    if variant == "original_clean":
        paths.update(
            {
                "detectgpt_gpt2_large": metrics_dir
                / "DetectGPT"
                / "metric_results"
                / "DetectGPT-GPT2Large-HC3-10000.json",
                "detectgpt_falcon_7b": metrics_dir
                / "DetectGPT"
                / "metric_results"
                / "DetectGPT-Falcon-HC3-10000.json",
                "detectgpt_falcon_instruct": metrics_dir
                / "DetectGPT"
                / "metric_results"
                / "DetectGPT-Falcon_Instruct-HC3-10000.json",
                "binoculars": metrics_dir / "Binoculars" / "Binoculars-Falcon-7bHC3-10000.json",
                "gptzero": metrics_dir
                / "metrics_XGB_SVM_GPTZERO"
                / "original_clean"
                / "hc3_unified_10000_seed42_clean_test_gptzero_like.json",
                "xgb": metrics_dir
                / "metrics_XGB_SVM_GPTZERO"
                / "original_clean"
                / "hc3_unified_10000_seed42_clean_test_xgboost_tfidf.json",
                "svm": metrics_dir
                / "metrics_XGB_SVM_GPTZERO"
                / "original_clean"
                / "hc3_unified_10000_seed42_clean_test_svm_tfidf.json",
            }
        )

    if variant == "stylistic_cleanup":
        paths.update(
            {
                "binoculars": metrics_dir / "Binoculars" / "Binoculars-stylistic-cleanup.json",
                "gptzero": metrics_dir
                / "metrics_XGB_SVM_GPTZERO"
                / "stylistic_cleanup"
                / "hc3_stylistic_cleanup_10000_clean_test_gptzero_like.json",
                "xgb": metrics_dir
                / "metrics_XGB_SVM_GPTZERO"
                / "stylistic_cleanup"
                / "hc3_stylistic_cleanup_10000_clean_test_xgboost_tfidf.json",
                "svm": metrics_dir
                / "metrics_XGB_SVM_GPTZERO"
                / "stylistic_cleanup"
                / "hc3_stylistic_cleanup_10000_clean_test_svm_tfidf.json",
            }
        )

    if variant == "recursive_paraphrase_depth_3":
        paths.update(
            {
                "gptzero": metrics_dir
                / "metrics_XGB_SVM_GPTZERO"
                / "recursive_paraphrase_depth_3"
                / "hc3_gpt_5_4_mini_recursive_paraphrase_depth_3_10000_clean_test_gptzero_like.json",
                "xgb": metrics_dir
                / "metrics_XGB_SVM_GPTZERO"
                / "recursive_paraphrase_depth_3"
                / "hc3_gpt_5_4_mini_recursive_paraphrase_depth_3_10000_clean_test_xgboost_tfidf.json",
                "svm": metrics_dir
                / "metrics_XGB_SVM_GPTZERO"
                / "recursive_paraphrase_depth_3"
                / "hc3_gpt_5_4_mini_recursive_paraphrase_depth_3_10000_clean_test_svm_tfidf.json",
            }
        )

    return paths


def load_rows(metrics_dir: Path, variants: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for variant in variants:
        paths = method_paths(metrics_dir, variant)
        for method in METHODS:
            path = paths.get(method.key)
            if path is None or not path.exists():
                continue
            payload = read_json(path)
            rows.append(
                {
                    "variant": variant,
                    "method_key": method.key,
                    "method": method.label,
                    "path": str(path),
                    "num_samples": int(payload.get("num_samples") or 0),
                    "tpr_at_0.1pct_fpr": metric_get(payload, "metrics_at_0.1pct_fpr", "tpr")
                    or metric_get(payload, "metrics_at_0.1pct_fpr", "recall"),
                    "f1_at_0.1pct_fpr": metric_get(payload, "metrics_at_0.1pct_fpr", "f1"),
                    "tpr_at_1pct_fpr": metric_get(payload, "metrics_at_1pct_fpr", "tpr")
                    or metric_get(payload, "metrics_at_1pct_fpr", "recall"),
                    "f1_at_1pct_fpr": metric_get(payload, "metrics_at_1pct_fpr", "f1"),
                }
            )
    return rows


def write_table(output_dir: Path, variants: List[str], rows: List[Dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    import csv

    suffix = variants[0] if len(variants) == 1 else "combined"
    out_path = output_dir / f"metrics_table_{suffix}.csv"
    fieldnames = [
        "variant",
        "method",
        "tpr_at_0.1pct_fpr",
        "f1_at_0.1pct_fpr",
        "tpr_at_1pct_fpr",
        "f1_at_1pct_fpr",
        "num_samples",
        "path",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return out_path


def plot_metric(
    output_dir: Path,
    variants: List[str],
    rows: List[Dict[str, Any]],
    active_methods: List[MethodSpec],
    metric_key: str,
    title: str,
    ylabel: str,
    filename_prefix: str,
) -> Path:
    import math

    import os
    # Avoid ~/.matplotlib permission issues in some environments (e.g., Colab/sandboxes).
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [m.label for m in active_methods]
    variant_to_method_value: Dict[str, Dict[str, Optional[float]]] = {}
    for variant in variants:
        variant_to_method_value[variant] = {m: None for m in methods}

    for row in rows:
        v = row["variant"]
        if v not in variant_to_method_value:
            continue
        variant_to_method_value[v][row["method"]] = row.get(metric_key)

    x = list(range(len(variants)))
    n_methods = len(methods)
    bar_w = 0.8 / max(1, n_methods)
    offsets = [(i - (n_methods - 1) / 2) * bar_w for i in range(n_methods)]

    fig_w = max(6.5, 1.2 * len(variants))
    fig, ax = plt.subplots(figsize=(fig_w, 4.2), dpi=160)

    # Deterministic palette keyed off the global METHODS list so colors stay consistent
    # across different plots, even when some methods are missing for a variant.
    cmap = plt.get_cmap("tab10")
    method_to_color = {m.label: cmap(i % 10) for i, m in enumerate(METHODS)}

    for i, method in enumerate(methods):
        xs = [xi + offsets[i] for xi in x]
        ys = [variant_to_method_value[v].get(method) for v in variants]
        plot_ys = [0.0 if (y is None or (isinstance(y, float) and math.isnan(y))) else float(y) for y in ys]
        bars = ax.bar(
            xs,
            plot_ys,
            width=bar_w,
            color=method_to_color.get(method, cmap(i % 10)),
            label=method,
        )
        for bar, y in zip(bars, ys):
            if y is None or (isinstance(y, float) and math.isnan(y)):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                float(y) + 0.01,
                f"{float(y):.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS.get(v, v.replace("_", " ")) for v in variants])
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()

    suffix = variants[0] if len(variants) == 1 else "combined"
    out_path = output_dir / f"{filename_prefix}_{suffix}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    rows = load_rows(args.metrics_dir, variants)

    if not rows:
        raise SystemExit(
            "No metrics JSONs were found for the requested variants. "
            "Check Metrics/ paths and the --variants value."
        )

    table_path = write_table(args.output_dir, variants, rows)
    print(f"Wrote {table_path}")

    # Use the same method ordering and color mapping across both plots.
    present_method_keys = {row["method_key"] for row in rows}
    active_methods = [m for m in METHODS if m.key in present_method_keys]

    tpr_plot = plot_metric(
        args.output_dir,
        variants,
        rows,
        active_methods,
        metric_key="tpr_at_0.1pct_fpr",
        title="TPR @ 0.1% FPR",
        ylabel="TPR @ 0.1% FPR",
        filename_prefix="tpr_at_0_1pct_fpr",
    )
    print(f"Wrote {tpr_plot}")

    f1_plot = plot_metric(
        args.output_dir,
        variants,
        rows,
        active_methods,
        metric_key="f1_at_0.1pct_fpr",
        title="F1 @ 0.1% FPR",
        ylabel="F1 @ 0.1% FPR",
        filename_prefix="f1_at_0_1pct_fpr",
    )
    print(f"Wrote {f1_plot}")


if __name__ == "__main__":
    main()
