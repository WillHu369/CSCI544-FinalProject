from __future__ import annotations

import ast
import json
import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gpt_zero.classical import BaselineTrainingConfig, ClassicalBaselineSuite, train_classical_baselines
from gpt_zero.gptzero_like import (
    CausalLMPerplexityScorer,
    FeatureExtractionConfig,
    GPTZeroLikeDetector,
    ScorerConfig,
    train_gptzero_like_detector,
)
from gpt_zero.hc3 import PrepareHC3Config, deduplicate_samples, normalize_hc3_rows, prepare_hc3_dataset
from gpt_zero.io_utils import dump_json, ensure_dir, read_table, reset_dir, timestamp_run_id, write_table
from gpt_zero.metrics import evaluate_predictions, fixed_fpr_metric_name
from gpt_zero.schemas import ensure_sample_schema
from gpt_zero.tfidf import TfidfFeatureConfig

VALID_SCORE_SPLITS = ("train", "val", "test")

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data" / "hc3"
BASELINE_MODEL_DIR = ARTIFACTS_DIR / "models" / "baselines"
GPTZERO_MODEL_DIR = ARTIFACTS_DIR / "models" / "gptzero_like"
BASELINE_RUN_DIR = ARTIFACTS_DIR / "runs" / "hc3_baselines_run"
GPTZERO_RUN_DIR = ARTIFACTS_DIR / "runs" / "hc3_gptzero_run"
REFERENCE_GPTZERO_METRICS_DIR = ARTIFACTS_DIR / "runs" / "hc3_gptzero_full_run" / "metrics"
DEFAULT_COLAB_TARGET_FPRS = (0.01, 0.001, 0.0001)
TEST_DATASET_DIR = PROJECT_ROOT / "test_dataset"
METRICS_SHARE_DIR = PROJECT_ROOT / "metrics_share"
TEST_DATASET_SAMPLE_DIR = ARTIFACTS_DIR / "data" / "test_dataset_samples"
TEST_DATASET_RUN_DIR = ARTIFACTS_DIR / "runs" / "test_dataset"
FILTERED_TRAINING_DATA_DIR = ARTIFACTS_DIR / "data" / "hc3_without_test_dataset"
HC3_UNIFIED_TEST_FILENAME = "hc3_unified_10000_seed42_clean_test.csv"
SHARED_METRIC_TARGETS = (
    (0.01, "metrics_at_1pct_fpr"),
    (0.001, "metrics_at_0.1pct_fpr"),
    (0.0001, "metrics_at_0.01pct_fpr"),
)


def _normalize_score_splits(values: list[str] | tuple[str, ...] | None) -> list[str]:
    if not values:
        return list(VALID_SCORE_SPLITS)

    normalized: list[str] = []
    for value in values:
        for item in str(value).split(","):
            candidate = item.strip().lower()
            if not candidate:
                continue
            if candidate not in VALID_SCORE_SPLITS:
                raise ValueError(f"Unsupported split '{candidate}'. Expected one of: {', '.join(VALID_SCORE_SPLITS)}")
            if candidate not in normalized:
                normalized.append(candidate)
    return normalized or list(VALID_SCORE_SPLITS)


def _resolve_split_path(
    data_dir: Path | str,
    split: str,
    split_paths: Mapping[str, Path | str] | None = None,
) -> Path:
    if split_paths and split in split_paths:
        explicit_path = Path(split_paths[split])
        if explicit_path.exists():
            return explicit_path
        raise FileNotFoundError(f"Explicit dataset split path for '{split}' does not exist: {explicit_path}")

    data_dir = Path(data_dir)
    candidates = (
        data_dir / f"{split}.parquet",
        data_dir / f"{split}.csv",
        data_dir / f"{split}.jsonl",
        data_dir / f"{split}.json",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find a dataset split for '{split}' under {data_dir}. "
        f"Expected one of: {', '.join(path.name for path in candidates)}"
    )


def find_split_path(
    data_dir: Path | str,
    split: str,
    split_paths: Mapping[str, Path | str] | None = None,
) -> Path:
    return _resolve_split_path(data_dir, split, split_paths)


def _clean_scalar(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _normalize_text_key(value: object) -> str:
    return " ".join(_clean_scalar(value).split())


def _prompt_key(domain: object, prompt: object) -> tuple[str, str]:
    return (_normalize_text_key(domain), _normalize_text_key(prompt))


def _decode_list_cell(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    for loader in (json.loads, ast.literal_eval):
        try:
            decoded = loader(text)
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(decoded, list):
            return [str(item) for item in decoded if str(item).strip()]
        return [str(decoded)]
    return [text]


def _dataset_name_from_path(path: Path | str) -> str:
    stem = Path(path).stem.lower()
    characters = [character if character.isalnum() else "_" for character in stem]
    slug = "".join(characters)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "dataset"


def _load_hc3_unified_rows(path: Path | str) -> list[dict]:
    source = Path(path)
    frame = pd.read_csv(source)
    required_columns = {"source", "question", "human_answers", "chatgpt_answers"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required HC3 unified columns: {missing}")

    rows: list[dict] = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            {
                **row,
                "source": _clean_scalar(row.get("source")),
                "question": _clean_scalar(row.get("question")),
                "human_answers": _decode_list_cell(row.get("human_answers")),
                "chatgpt_answers": _decode_list_cell(row.get("chatgpt_answers")),
            }
        )
    return rows


def _candidate_hc3_unified_sources() -> list[Path]:
    return [
        PROJECT_ROOT / "HC3-Dataset-Samples" / "hc3_unified_10000_seed42_clean.csv",
        PROJECT_ROOT.parent / "HC3-Dataset-Samples" / "hc3_unified_10000_seed42_clean.csv",
        PROJECT_ROOT.parent.parent / "HC3-Dataset-Samples" / "hc3_unified_10000_seed42_clean.csv",
    ]


def ensure_hc3_unified_test_file(
    *,
    source_file: Path | str | None = None,
    test_dataset_dir: Path | str = TEST_DATASET_DIR,
    overwrite: bool = False,
) -> Path:
    destination = ensure_dir(test_dataset_dir) / HC3_UNIFIED_TEST_FILENAME
    if destination.exists() and not overwrite:
        return destination

    candidates = [Path(source_file)] if source_file else []
    candidates.extend(_candidate_hc3_unified_sources())
    for candidate in candidates:
        if candidate.exists():
            shutil.copy2(candidate, destination)
            return destination
    raise FileNotFoundError(
        "Could not find hc3_unified_10000_seed42_clean.csv to copy into test_dataset. "
        f"Checked: {', '.join(str(path) for path in candidates)}"
    )


def _heldout_prompt_keys_from_test_dataset(test_dataset_dir: Path | str) -> set[tuple[str, str]]:
    dataset_dir = Path(test_dataset_dir)
    keys: set[tuple[str, str]] = set()
    for csv_path in sorted(dataset_dir.glob("*.csv")):
        frame = pd.read_csv(csv_path, usecols=lambda column: column in {"source", "question"})
        if {"source", "question"}.difference(frame.columns):
            continue
        keys.update(_prompt_key(row["source"], row["question"]) for row in frame.to_dict(orient="records"))
    if not keys:
        raise RuntimeError(f"No HC3 unified test prompts were found in {dataset_dir}")
    return keys


def prepare_training_data_without_test_dataset(
    *,
    data_dir: Path | str = DATA_DIR,
    test_dataset_dir: Path | str = TEST_DATASET_DIR,
    output_dir: Path | str = FILTERED_TRAINING_DATA_DIR,
    filter_splits: tuple[str, ...] = ("train", "val"),
) -> dict:
    heldout_keys = _heldout_prompt_keys_from_test_dataset(test_dataset_dir)
    output = reset_dir(output_dir, preserve_names=(".gitkeep",))
    split_paths: dict[str, str] = {}
    split_counts: dict[str, dict[str, int]] = {}

    for split in VALID_SCORE_SPLITS:
        try:
            source_path = _resolve_split_path(data_dir, split)
        except FileNotFoundError:
            continue

        frame = read_table(source_path)
        original_rows = int(len(frame))
        if split in filter_splits:
            row_keys = frame.apply(lambda row: _prompt_key(row.get("domain"), row.get("prompt")), axis=1)
            keep_mask = ~row_keys.isin(heldout_keys)
            frame = frame.loc[keep_mask].reset_index(drop=True)
        else:
            frame = frame.reset_index(drop=True)

        if split in filter_splits and frame.empty and original_rows > 0:
            raise RuntimeError(
                f"Filtering test_dataset prompts removed every row from the {split} split. "
                "Use a larger non-overlapping training source before retraining detectors."
            )

        destination = output / f"{split}.csv"
        write_table(frame, destination)
        split_paths[split] = str(destination)
        split_counts[split] = {
            "source_rows": original_rows,
            "output_rows": int(len(frame)),
            "removed_rows": original_rows - int(len(frame)),
        }

    manifest = {
        "data_dir": str(data_dir),
        "test_dataset_dir": str(test_dataset_dir),
        "heldout_prompt_count": len(heldout_keys),
        "filter_splits": list(filter_splits),
        "split_counts": split_counts,
        "split_paths": split_paths,
    }
    dump_json(manifest, output / "manifest.json")
    return manifest


def prepare_test_dataset_files(
    *,
    test_dataset_dir: Path | str = TEST_DATASET_DIR,
    output_dir: Path | str = TEST_DATASET_SAMPLE_DIR,
) -> dict:
    test_dataset_dir = Path(test_dataset_dir)
    output = reset_dir(output_dir, preserve_names=(".gitkeep",))
    datasets: list[dict] = []

    for compact_path in sorted(test_dataset_dir.glob("*.csv")):
        dataset_name = _dataset_name_from_path(compact_path)
        rows = _load_hc3_unified_rows(compact_path)
        sample_frame = normalize_hc3_rows(rows, dataset_name=dataset_name)
        sample_frame, deduplication = deduplicate_samples(sample_frame)
        sample_frame = ensure_sample_schema(sample_frame)
        sample_frame["split"] = "test"
        sample_frame = sample_frame.sort_values(["domain", "label", "sample_id"]).reset_index(drop=True)

        dataset_dir = ensure_dir(output / dataset_name)
        sample_path = dataset_dir / "test.csv"
        write_table(sample_frame, sample_path)
        datasets.append(
            {
                "dataset_name": dataset_name,
                "dataset_used": compact_path.name,
                "compact_path": str(compact_path),
                "sample_path": str(sample_path),
                "num_samples": int(len(sample_frame)),
                "deduplication": deduplication,
            }
        )

    if not datasets:
        raise FileNotFoundError(f"No CSV test datasets found in {test_dataset_dir}")

    manifest = {
        "test_dataset_dir": str(test_dataset_dir),
        "output_dir": str(output),
        "datasets": datasets,
    }
    dump_json(manifest, output / "manifest.json")
    return manifest


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _shared_model_metadata(detector_name: str, config: ColabExperimentConfig | None) -> tuple[str, list[str]]:
    if detector_name == "gptzero_like":
        additional_models = [str(config.lm_model)] if config is not None else []
        return "gptzero_like_logistic_regression", additional_models
    if detector_name == "svm_tfidf":
        return "linear_svm_tfidf", []
    if detector_name == "xgboost_tfidf":
        return "xgboost_tfidf", []
    return detector_name, []


def _shared_metrics_block(row: pd.Series, target_fpr: float) -> dict[str, float]:
    return {
        "f1": _finite_float(row.get(fixed_fpr_metric_name("f1", target_fpr))),
        "accuracy": _finite_float(row.get(fixed_fpr_metric_name("accuracy", target_fpr))),
        "precision": _finite_float(row.get(fixed_fpr_metric_name("precision", target_fpr))),
        "recall": _finite_float(row.get(fixed_fpr_metric_name("recall", target_fpr))),
        "auc_roc": _finite_float(row.get("roc_auc")),
    }


def export_metrics_share_json(
    *,
    metrics_dir: Path | str,
    metrics_share_dir: Path | str = METRICS_SHARE_DIR,
    experiment_name: str,
    dataset_used: str,
    config: ColabExperimentConfig | None = None,
) -> list[dict]:
    metrics_dir = Path(metrics_dir)
    destination = ensure_dir(metrics_share_dir)
    summary_path = metrics_dir / "metrics_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing metrics summary: {summary_path}")

    summary = pd.read_csv(summary_path)
    summary = summary.loc[summary["split"].astype(str) == "test"].copy()
    exported: list[dict] = []
    for _, row in summary.iterrows():
        detector_name = str(row["detector_name"])
        model_used, additional_models = _shared_model_metadata(detector_name, config)
        payload = {
            "experiment_name": f"{experiment_name}_{detector_name}",
            "detection_method": detector_name,
            "model_used": model_used,
            "dataset_used": dataset_used,
            "num_samples": int(row.get("num_samples", 0) or 0),
            "additional_details": {
                "additional_models_used": additional_models,
                "notes": "Exported from ZeroGPT Colab evaluation on test_dataset.",
            },
        }
        for target_fpr, key in SHARED_METRIC_TARGETS:
            payload[key] = _shared_metrics_block(row, target_fpr)

        output_path = destination / f"{experiment_name}_{detector_name}.json"
        dump_json(payload, output_path)
        exported.append({"path": str(output_path), "payload": payload})
    return exported


@dataclass
class ColabExperimentConfig:
    data_dir: Path = DATA_DIR
    split_paths: Mapping[str, Path | str] | None = None
    baseline_model_dir: Path = BASELINE_MODEL_DIR
    gptzero_model_dir: Path = GPTZERO_MODEL_DIR
    baseline_run_dir: Path = BASELINE_RUN_DIR
    gptzero_run_dir: Path = GPTZERO_RUN_DIR
    lm_model: str = "gpt2"
    device: str = "cuda"
    local_files_only: bool = False
    row_batch_size: int = 64
    perplexity_batch_size: int = 16
    stride: int = 512
    max_length: int | None = None
    max_sentences_per_text: int | None = None
    score_splits: tuple[str, ...] = ("test",)
    target_fpr: float = 0.01
    target_fprs: tuple[float, ...] = DEFAULT_COLAB_TARGET_FPRS
    word_max_features: int = 20000
    char_max_features: int = 15000
    min_df: int = 2
    svm_c: float = 1.0
    xgb_batch_size: int = 1024
    xgb_estimators: int = 120
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_device: str = "cuda"
    xgb_early_stopping_rounds: int = 20
    xgb_eval_log_interval: int = 10


def project_paths() -> dict[str, str]:
    return {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "baseline_model_dir": str(BASELINE_MODEL_DIR),
        "gptzero_model_dir": str(GPTZERO_MODEL_DIR),
        "baseline_run_dir": str(BASELINE_RUN_DIR),
        "gptzero_run_dir": str(GPTZERO_RUN_DIR),
        "reference_gptzero_metrics_dir": str(REFERENCE_GPTZERO_METRICS_DIR),
        "test_dataset_dir": str(TEST_DATASET_DIR),
        "metrics_share_dir": str(METRICS_SHARE_DIR),
        "filtered_training_data_dir": str(FILTERED_TRAINING_DATA_DIR),
        "notebook": str(PROJECT_ROOT / "hc3_colab_workflow.ipynb"),
    }


def load_hc3_manifest() -> dict:
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No HC3 manifest found at {manifest_path}")
    import json

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def rebuild_hc3_from_source(
    output_dir: Path | str,
    *,
    input_file: Path | str | None = None,
    hf_dataset: str = "Hello-SimpleAI/HC3",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    max_samples_per_label: int | None = None,
) -> dict:
    config = PrepareHC3Config(
        output_dir=Path(output_dir),
        input_file=Path(input_file) if input_file else None,
        hf_dataset=hf_dataset,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        max_samples_per_label=max_samples_per_label,
        group_by_prompt=True,
        deduplicate_texts=True,
    )
    return prepare_hc3_dataset(config)


def train_baselines(config: ColabExperimentConfig) -> dict:
    feature_config = TfidfFeatureConfig(
        word_max_features=config.word_max_features,
        char_max_features=config.char_max_features,
        min_df=config.min_df,
    )
    training_config = BaselineTrainingConfig(
        svm_c=config.svm_c,
        batch_size=config.row_batch_size,
        xgb_batch_size=config.xgb_batch_size,
        xgb_estimators=config.xgb_estimators,
        xgb_max_depth=config.xgb_max_depth,
        xgb_learning_rate=config.xgb_learning_rate,
        xgb_subsample=config.xgb_subsample,
        xgb_colsample_bytree=config.xgb_colsample_bytree,
        xgb_device=config.xgb_device,
        xgb_early_stopping_rounds=config.xgb_early_stopping_rounds,
        xgb_eval_log_interval=config.xgb_eval_log_interval,
    )
    train_source = _resolve_split_path(config.data_dir, "train", config.split_paths)
    try:
        val_source = _resolve_split_path(config.data_dir, "val", config.split_paths)
    except FileNotFoundError:
        if config.split_paths and "val" in config.split_paths:
            raise
        val_source = None
    return train_classical_baselines(
        train_source=train_source,
        model_dir=config.baseline_model_dir,
        feature_config=feature_config,
        training_config=training_config,
        val_source=val_source,
    )


def train_gptzero(config: ColabExperimentConfig) -> dict:
    scorer_config = ScorerConfig(
        model_name=config.lm_model,
        device=config.device,
        stride=config.stride,
        max_length=config.max_length,
        local_files_only=config.local_files_only,
        perplexity_batch_size=config.perplexity_batch_size,
    )
    feature_config = FeatureExtractionConfig(max_sentences_per_text=config.max_sentences_per_text)
    train_source = _resolve_split_path(config.data_dir, "train", config.split_paths)
    try:
        val_source = _resolve_split_path(config.data_dir, "val", config.split_paths)
    except FileNotFoundError:
        if config.split_paths and "val" in config.split_paths:
            raise
        val_source = None
    return train_gptzero_like_detector(
        train_source=train_source,
        val_source=val_source,
        model_dir=config.gptzero_model_dir,
        scorer_config=scorer_config,
        feature_config=feature_config,
        batch_size=config.row_batch_size,
    )


def score_models(
    *,
    data_dir: Path | str = DATA_DIR,
    split_paths: Mapping[str, Path | str] | None = None,
    run_dir: Path | str,
    run_id: str | None = None,
    baseline_model_dir: Path | str | None = None,
    gptzero_model_dir: Path | str | None = None,
    batch_size: int = 64,
    score_splits: tuple[str, ...] = ("test",),
) -> dict:
    data_dir = Path(data_dir)
    run_dir = reset_dir(run_dir, preserve_names=(".gitkeep",))
    predictions_dir = ensure_dir(run_dir / "predictions")
    requested_splits = _normalize_score_splits(score_splits)
    run_id = run_id or timestamp_run_id("colab_run")

    baselines_suite = None
    if baseline_model_dir is not None and Path(baseline_model_dir).exists():
        baselines_suite = ClassicalBaselineSuite.load(baseline_model_dir)

    gptzero_detector = None
    gptzero_scorer = None
    gptzero_cache_dir = None
    if gptzero_model_dir is not None and (Path(gptzero_model_dir) / "gptzero_like.joblib").exists():
        gptzero_detector = GPTZeroLikeDetector.load(gptzero_model_dir)
        gptzero_scorer = CausalLMPerplexityScorer(gptzero_detector.scorer_config)
        gptzero_cache_dir = Path(gptzero_model_dir) / "feature_cache"

    if baselines_suite is None and gptzero_detector is None:
        raise RuntimeError("No detectors were found. Provide a trained baselines directory and/or GPTZero-like model directory.")

    scored_splits: list[str] = []
    for split in requested_splits:
        try:
            split_path = _resolve_split_path(data_dir, split, split_paths)
        except FileNotFoundError:
            if split_paths and split in split_paths:
                raise
            continue

        split_predictions = []
        if gptzero_detector is not None:
            feature_frame = gptzero_detector.build_feature_frame(
                split_path,
                scorer=gptzero_scorer,
                batch_size=batch_size,
                progress_label=f"gptzero-{split}",
                cache_dir=gptzero_cache_dir,
            )
            split_predictions.append(gptzero_detector.predict_from_features(feature_frame, run_id))

        if baselines_suite is not None:
            split_predictions.append(baselines_suite.predict(split_path, run_id, batch_size=batch_size))

        combined = pd.concat(split_predictions, ignore_index=True)
        write_table(combined, predictions_dir / f"{split}.parquet")
        scored_splits.append(split)

    dump_json(
        {
            "run_id": run_id,
            "data_dir": str(data_dir),
            "split_paths": {key: str(value) for key, value in split_paths.items()} if split_paths else None,
            "baseline_model_dir": str(baseline_model_dir) if baseline_model_dir else None,
            "gptzero_model_dir": str(gptzero_model_dir) if gptzero_model_dir else None,
            "score_splits": scored_splits,
        },
        run_dir / "run_config.json",
    )
    return {
        "run_id": run_id,
        "predictions_dir": str(predictions_dir),
        "score_splits": scored_splits,
    }


def evaluate_run(
    *,
    data_dir: Path | str = DATA_DIR,
    split_paths: Mapping[str, Path | str] | None = None,
    predictions_dir: Path | str,
    output_dir: Path | str,
    target_fpr: float = 0.01,
    target_fprs: tuple[float, ...] | None = None,
) -> dict:
    sample_frames = []
    prediction_frames = []
    for split in VALID_SCORE_SPLITS:
        try:
            sample_path = _resolve_split_path(data_dir, split, split_paths)
        except FileNotFoundError:
            if split_paths and split in split_paths:
                raise
            sample_path = None
        prediction_path = Path(predictions_dir) / f"{split}.parquet"
        if sample_path is not None and sample_path.exists():
            sample_frames.append(read_table(sample_path))
        if prediction_path.exists():
            prediction_frames.append(read_table(prediction_path))

    if not sample_frames:
        raise RuntimeError(f"No dataset splits found under {data_dir}")
    if not prediction_frames:
        raise RuntimeError(f"No prediction files found under {predictions_dir}")

    samples = pd.concat(sample_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return evaluate_predictions(samples, predictions, output_dir, target_fpr=target_fpr, target_fprs=target_fprs)


def run_test_dataset_evaluations(
    *,
    config: ColabExperimentConfig,
    test_dataset_dir: Path | str = TEST_DATASET_DIR,
    prepared_data_dir: Path | str = TEST_DATASET_SAMPLE_DIR,
    run_root: Path | str = TEST_DATASET_RUN_DIR,
    metrics_share_dir: Path | str = METRICS_SHARE_DIR,
    batch_size: int | None = None,
    reset_metrics_share: bool = True,
) -> dict:
    prepared_manifest = prepare_test_dataset_files(
        test_dataset_dir=test_dataset_dir,
        output_dir=prepared_data_dir,
    )
    run_root = reset_dir(run_root, preserve_names=(".gitkeep",))
    metrics_share_dir = reset_dir(metrics_share_dir, preserve_names=(".gitkeep",)) if reset_metrics_share else ensure_dir(metrics_share_dir)

    exported_json: list[dict] = []
    summary_frames: list[pd.DataFrame] = []
    roc_frames: list[pd.DataFrame] = []
    dataset_results: list[dict] = []

    for dataset_info in prepared_manifest["datasets"]:
        dataset_name = dataset_info["dataset_name"]
        dataset_used = dataset_info["dataset_used"]
        sample_path = Path(dataset_info["sample_path"])
        dataset_run_dir = run_root / dataset_name

        scoring = score_models(
            data_dir=sample_path.parent,
            split_paths={"test": sample_path},
            run_dir=dataset_run_dir,
            baseline_model_dir=config.baseline_model_dir,
            gptzero_model_dir=config.gptzero_model_dir,
            batch_size=batch_size or config.row_batch_size,
            score_splits=("test",),
        )
        metrics_dir = dataset_run_dir / "metrics"
        evaluation = evaluate_run(
            data_dir=sample_path.parent,
            split_paths={"test": sample_path},
            predictions_dir=Path(scoring["predictions_dir"]),
            output_dir=metrics_dir,
            target_fpr=config.target_fpr,
            target_fprs=config.target_fprs,
        )
        shared_exports = export_metrics_share_json(
            metrics_dir=metrics_dir,
            metrics_share_dir=metrics_share_dir,
            experiment_name=dataset_name,
            dataset_used=dataset_used,
            config=config,
        )
        exported_json.extend(shared_exports)

        summary_path = metrics_dir / "metrics_summary.csv"
        roc_path = metrics_dir / "roc_points.csv"
        if summary_path.exists():
            summary_frame = pd.read_csv(summary_path)
            summary_frame["dataset_name"] = dataset_name
            summary_frame["dataset_used"] = dataset_used
            summary_frames.append(summary_frame)
        if roc_path.exists():
            roc_frame = pd.read_csv(roc_path)
            roc_frame["dataset_name"] = dataset_name
            roc_frame["dataset_used"] = dataset_used
            roc_frames.append(roc_frame)

        dataset_results.append(
            {
                "dataset_name": dataset_name,
                "dataset_used": dataset_used,
                "sample_path": str(sample_path),
                "run_dir": str(dataset_run_dir),
                "metrics_dir": str(metrics_dir),
                "scoring": scoring,
                "evaluation": evaluation,
                "shared_metric_files": [item["path"] for item in shared_exports],
            }
        )

    combined_summary_path = metrics_share_dir / "all_test_dataset_metrics_summary.csv"
    combined_roc_path = metrics_share_dir / "all_test_dataset_roc_points.csv"
    if summary_frames:
        write_table(pd.concat(summary_frames, ignore_index=True), combined_summary_path)
    if roc_frames:
        write_table(pd.concat(roc_frames, ignore_index=True), combined_roc_path)

    manifest = {
        "test_dataset_dir": str(test_dataset_dir),
        "prepared_data_dir": str(prepared_data_dir),
        "run_root": str(run_root),
        "metrics_share_dir": str(metrics_share_dir),
        "combined_summary_path": str(combined_summary_path),
        "combined_roc_path": str(combined_roc_path),
        "datasets": dataset_results,
        "shared_metric_files": [item["path"] for item in exported_json],
    }
    dump_json(manifest, metrics_share_dir / "manifest.json")
    return manifest


def run_baseline_reference(
    *,
    config: ColabExperimentConfig | None = None,
    retrain: bool = False,
    score_splits: tuple[str, ...] = ("test",),
) -> dict:
    config = config or ColabExperimentConfig()
    if retrain or not (Path(config.baseline_model_dir) / "metadata.json").exists():
        training = train_baselines(config)
    else:
        training = {"used_existing_models": True, "model_dir": str(config.baseline_model_dir)}

    try:
        scoring = score_models(
            data_dir=config.data_dir,
            split_paths=config.split_paths,
            run_dir=config.baseline_run_dir,
            baseline_model_dir=config.baseline_model_dir,
            gptzero_model_dir=None,
            batch_size=config.row_batch_size,
            score_splits=score_splits,
        )
    except Exception as exc:
        message = str(exc)
        if not retrain and (
            "InconsistentVersionWarning" in message
            or "multi_class" in message
            or "unpickle estimator" in message
        ):
            raise RuntimeError(
                "The saved baseline scikit-learn artifacts are incompatible with the current Colab "
                "environment. Reinstall the pinned requirements and rerun the notebook from the top. "
                "If the mismatch persists, run the baseline section with retrain=True so fresh models "
                "are created in the current environment."
            ) from exc
        raise

    evaluation = evaluate_run(
        data_dir=config.data_dir,
        split_paths=config.split_paths,
        predictions_dir=Path(scoring["predictions_dir"]),
        output_dir=Path(config.baseline_run_dir) / "metrics",
        target_fpr=config.target_fpr,
        target_fprs=config.target_fprs,
    )
    return {"training": training, "scoring": scoring, "evaluation": evaluation}


def run_gptzero_experiment(
    *,
    config: ColabExperimentConfig | None = None,
    train_model: bool = True,
    score_splits: tuple[str, ...] = ("test",),
) -> dict:
    config = config or ColabExperimentConfig()
    if train_model:
        training = train_gptzero(config)
    else:
        training = {"used_existing_model": True, "model_dir": str(config.gptzero_model_dir)}

    scoring = score_models(
        data_dir=config.data_dir,
        split_paths=config.split_paths,
        run_dir=config.gptzero_run_dir,
        baseline_model_dir=None,
        gptzero_model_dir=config.gptzero_model_dir,
        batch_size=config.row_batch_size,
        score_splits=score_splits,
    )
    evaluation = evaluate_run(
        data_dir=config.data_dir,
        split_paths=config.split_paths,
        predictions_dir=Path(scoring["predictions_dir"]),
        output_dir=Path(config.gptzero_run_dir) / "metrics",
        target_fpr=config.target_fpr,
        target_fprs=config.target_fprs,
    )
    return {"training": training, "scoring": scoring, "evaluation": evaluation}


def load_metrics(metrics_dir: Path | str) -> dict[str, pd.DataFrame]:
    metrics_dir = Path(metrics_dir)
    outputs: dict[str, pd.DataFrame] = {}
    for name in ("metrics_summary.csv", "metrics_by_domain.csv", "roc_points.csv"):
        path = metrics_dir / name
        outputs[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return outputs


def load_reference_metrics() -> dict[str, dict[str, pd.DataFrame]]:
    return {
        "baseline": load_metrics(BASELINE_RUN_DIR / "metrics"),
        "gptzero": load_metrics(REFERENCE_GPTZERO_METRICS_DIR),
    }


def compare_reference_summaries() -> pd.DataFrame:
    baseline_summary = pd.read_csv(BASELINE_RUN_DIR / "metrics" / "metrics_summary.csv")
    gptzero_summary = pd.read_csv(REFERENCE_GPTZERO_METRICS_DIR / "metrics_summary.csv")
    combined = pd.concat([baseline_summary, gptzero_summary], ignore_index=True)
    return combined.sort_values(["split", "detector_name"]).reset_index(drop=True)


__all__ = [
    "ARTIFACTS_DIR",
    "BASELINE_MODEL_DIR",
    "BASELINE_RUN_DIR",
    "PROJECT_ROOT",
    "ColabExperimentConfig",
    "DATA_DIR",
    "DEFAULT_COLAB_TARGET_FPRS",
    "FILTERED_TRAINING_DATA_DIR",
    "GPTZERO_MODEL_DIR",
    "GPTZERO_RUN_DIR",
    "METRICS_SHARE_DIR",
    "REFERENCE_GPTZERO_METRICS_DIR",
    "TEST_DATASET_DIR",
    "TEST_DATASET_RUN_DIR",
    "TEST_DATASET_SAMPLE_DIR",
    "project_paths",
    "compare_reference_summaries",
    "ensure_hc3_unified_test_file",
    "evaluate_run",
    "export_metrics_share_json",
    "find_split_path",
    "load_hc3_manifest",
    "load_metrics",
    "load_reference_metrics",
    "prepare_test_dataset_files",
    "prepare_training_data_without_test_dataset",
    "rebuild_hc3_from_source",
    "run_baseline_reference",
    "run_gptzero_experiment",
    "run_test_dataset_evaluations",
    "score_models",
    "train_baselines",
    "train_gptzero",
]
