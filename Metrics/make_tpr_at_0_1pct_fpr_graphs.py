import os
# Avoid matplotlib trying to write to ~/.matplotlib in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

#!/usr/bin/env python3

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DetectorSpec:
    key: str
    label: str
    color: str


DETECTORS = [
    DetectorSpec("binoculars", "Binoculars", "#c44e52"),
    DetectorSpec("detectgpt_gpt2", "DetectGPT (GPT-2)", "#4c78a8"),
    DetectorSpec("detectgpt_falcon", "DetectGPT (Falcon-7B)", "#f58518"),
    DetectorSpec("detectgpt_falcon_instruct", "DetectGPT (Falcon-Instruct)", "#54a24b"),
    DetectorSpec("gptzero", "GPTZero", "#b79a56"),
    DetectorSpec("xgb", "XGB", "#72b7b2"),
    DetectorSpec("svm", "SVM", "#9d755d"),
]


DATASET_GROUPS = [("original_clean", "Original clean")]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _metric_at_0_1pct_fpr(metrics_obj: dict, metric_key: str) -> Optional[float]:
    if not isinstance(metrics_obj, dict):
        return None
    if metric_key == "tpr":
        # Several files store TPR redundantly as recall and/or tpr.
        for key in ("tpr", "recall", "tpr_at_0_1pct_fpr"):
            value = metrics_obj.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        return None

    value = metrics_obj.get(metric_key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_from_metrics_json(path: Path, metric_key: str) -> Optional[float]:
    data = _load_json(path)
    metrics = data.get("metrics_at_0.1pct_fpr")
    return _metric_at_0_1pct_fpr(metrics, metric_key)


def _extract_xgb_svm_gptzero_metrics(group_dir: Path) -> dict[str, dict[str, Optional[float]]]:
    out: dict[str, dict[str, Optional[float]]] = {
        "gptzero": {"tpr": None, "f1": None},
        "xgb": {"tpr": None, "f1": None},
        "svm": {"tpr": None, "f1": None},
    }
    if not group_dir.exists():
        return out

    for json_path in sorted(group_dir.glob("*.json")):
        data = _load_json(json_path)
        method = str(data.get("detection_method") or "").lower()
        metrics = data.get("metrics_at_0.1pct_fpr")
        tpr = _metric_at_0_1pct_fpr(metrics, "tpr")
        f1 = _metric_at_0_1pct_fpr(metrics, "f1")

        if method == "gptzero_like":
            out["gptzero"] = {"tpr": tpr, "f1": f1}
        elif method == "xgboost_tfidf":
            out["xgb"] = {"tpr": tpr, "f1": f1}
        elif method == "svm_tfidf":
            out["svm"] = {"tpr": tpr, "f1": f1}

    return out


def build_table() -> pd.DataFrame:
    rows = []

    # Per-dataset values for gptzero/xgb/svm.
    baseline_dir = ROOT / "metrics_XGB_SVM_GPTZERO"
    group_values: dict[str, dict[str, dict[str, Optional[float]]]] = {}
    for group_key, _ in DATASET_GROUPS:
        group_values[group_key] = _extract_xgb_svm_gptzero_metrics(baseline_dir / group_key)

    # DetectGPT + Binoculars only exist (currently) for the "original clean" run.
    det_gpt2 = ROOT / "DetectGPT" / "metric_results" / "DetectGPT-GPT2Large-HC3-10000.json"
    det_falcon = ROOT / "DetectGPT" / "metric_results" / "DetectGPT-Falcon-HC3-10000.json"
    det_falcon_inst = (
        ROOT / "DetectGPT" / "metric_results" / "DetectGPT-Falcon_Instruct-HC3-10000.json"
    )
    binoculars = ROOT / "Binoculars" / "Binoculars-Falcon-7bHC3-10000.json"

    original_only = {
        "binoculars": {
            "tpr": _extract_from_metrics_json(binoculars, "tpr") if binoculars.exists() else None,
            "f1": _extract_from_metrics_json(binoculars, "f1") if binoculars.exists() else None,
        },
        "detectgpt_gpt2": {
            "tpr": _extract_from_metrics_json(det_gpt2, "tpr") if det_gpt2.exists() else None,
            "f1": _extract_from_metrics_json(det_gpt2, "f1") if det_gpt2.exists() else None,
        },
        "detectgpt_falcon": {
            "tpr": _extract_from_metrics_json(det_falcon, "tpr") if det_falcon.exists() else None,
            "f1": _extract_from_metrics_json(det_falcon, "f1") if det_falcon.exists() else None,
        },
        "detectgpt_falcon_instruct": {
            "tpr": _extract_from_metrics_json(det_falcon_inst, "tpr")
            if det_falcon_inst.exists()
            else None,
            "f1": _extract_from_metrics_json(det_falcon_inst, "f1")
            if det_falcon_inst.exists()
            else None,
        },
    }

    for group_key, group_label in DATASET_GROUPS:
        for det in DETECTORS:
            tpr = None
            f1 = None
            if det.key in ("gptzero", "xgb", "svm"):
                tpr = group_values[group_key].get(det.key, {}).get("tpr")
                f1 = group_values[group_key].get(det.key, {}).get("f1")
            else:
                # Only populated for original_clean.
                if group_key == "original_clean":
                    tpr = original_only.get(det.key, {}).get("tpr")
                    f1 = original_only.get(det.key, {}).get("f1")

            rows.append(
                {
                    "dataset_group": group_key,
                    "dataset_label": group_label,
                    "detector_key": det.key,
                    "detector_label": det.label,
                    "tpr_at_0_1pct_fpr": tpr,
                    "f1_at_0_1pct_fpr": f1,
                }
            )

    return pd.DataFrame(rows)


def plot_grouped_bars(
    df: pd.DataFrame,
    metric_column: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> None:
    df = df.copy()
    df[metric_column] = pd.to_numeric(df[metric_column], errors="coerce")

    groups = [label for _, label in DATASET_GROUPS]
    dets = DETECTORS

    x = list(range(len(groups)))
    width = 0.11
    offsets = [(i - (len(dets) - 1) / 2.0) * width for i in range(len(dets))]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    for i, det in enumerate(dets):
        vals = []
        is_missing = []
        for _, group_label in DATASET_GROUPS:
            v = df.loc[
                (df["dataset_label"] == group_label) & (df["detector_key"] == det.key),
                metric_column,
            ]
            value = float(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else 0.0
            missing = not (len(v) and pd.notna(v.iloc[0]))
            vals.append(value)
            is_missing.append(missing)

        bar_x = [xi + offsets[i] for xi in x]
        bars = ax.bar(
            bar_x,
            vals,
            width=width,
            label=det.label,
            color=det.color,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.3,
        )

        for b, v, missing in zip(bars, vals, is_missing):
            if missing:
                b.set_alpha(0.15)
                b.set_hatch("//")
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    0.015,
                    "NA",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#555555",
                    rotation=90,
                )
            else:
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    min(0.98, v + 0.02),
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#111111",
                )

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_title(title)

    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.18))
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = build_table()
    plots_dir = ROOT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(plots_dir / "metrics_at_0_1pct_fpr_original_clean_table.csv", index=False)
    plot_grouped_bars(
        df,
        metric_column="tpr_at_0_1pct_fpr",
        y_label="TPR @ 0.1% FPR",
        title="TPR @ 0.1% FPR (FPR=0.001)",
        out_path=plots_dir / "tpr_at_0_1pct_fpr_original_clean.png",
    )
    plot_grouped_bars(
        df,
        metric_column="f1_at_0_1pct_fpr",
        y_label="F1 @ 0.1% FPR",
        title="F1 @ 0.1% FPR (FPR=0.001)",
        out_path=plots_dir / "f1_at_0_1pct_fpr_original_clean.png",
    )

    print(f"Wrote {plots_dir / 'metrics_at_0_1pct_fpr_original_clean_table.csv'}")
    print(f"Wrote {plots_dir / 'tpr_at_0_1pct_fpr_original_clean.png'}")
    print(f"Wrote {plots_dir / 'f1_at_0_1pct_fpr_original_clean.png'}")


if __name__ == "__main__":
    main()
