from __future__ import annotations

from typing import Iterable

import pandas as pd

LABEL_HUMAN = "human"
LABEL_AI = "ai"
LABEL_TO_ID = {LABEL_HUMAN: 0, LABEL_AI: 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}

SAMPLE_COLUMNS = [
    "sample_id",
    "dataset",
    "domain",
    "prompt",
    "text",
    "label",
    "source_model",
    "split",
    "variant_type",
    "parent_sample_id",
    "attack_name",
    "attack_metadata",
]

PREDICTION_COLUMNS = [
    "run_id",
    "detector_name",
    "sample_id",
    "score",
    "prob_ai",
    "pred_label",
]

GPTZERO_DIAGNOSTIC_COLUMNS = [
    "doc_perplexity",
    "sentence_perplexity_mean",
    "sentence_perplexity_std",
    "burstiness",
]

CLASSICAL_DIAGNOSTIC_COLUMNS = ["margin"]


def coerce_label(value: object) -> str:
    if value is None:
        raise ValueError("Label cannot be None.")

    normalized = str(value).strip().lower()
    if normalized in {"human", "human_only", "0", "real"}:
        return LABEL_HUMAN
    if normalized in {"ai", "machine", "chatgpt", "gpt", "1", "llm"}:
        return LABEL_AI
    raise ValueError(f"Unsupported label value: {value!r}")


def ensure_columns(frame: pd.DataFrame, expected: Iterable[str], frame_name: str) -> None:
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def ensure_sample_schema(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    defaults = {
        "variant_type": "original",
        "parent_sample_id": None,
        "attack_name": None,
        "attack_metadata": "{}",
    }
    for column in SAMPLE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = defaults.get(column)

    normalized["label"] = normalized["label"].map(coerce_label)
    return normalized[SAMPLE_COLUMNS]
