from __future__ import annotations

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
from gpt_zero.hc3 import PrepareHC3Config, prepare_hc3_dataset
from gpt_zero.io_utils import dump_json, ensure_dir, read_table, timestamp_run_id, write_table
from gpt_zero.metrics import evaluate_predictions
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


def _resolve_split_path(data_dir: Path | str, split: str) -> Path:
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


def find_split_path(data_dir: Path | str, split: str) -> Path:
    return _resolve_split_path(data_dir, split)


@dataclass
class ColabExperimentConfig:
    data_dir: Path = DATA_DIR
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
    train_source = _resolve_split_path(config.data_dir, "train")
    try:
        val_source = _resolve_split_path(config.data_dir, "val")
    except FileNotFoundError:
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
    train_source = _resolve_split_path(config.data_dir, "train")
    try:
        val_source = _resolve_split_path(config.data_dir, "val")
    except FileNotFoundError:
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
    run_dir: Path | str,
    run_id: str | None = None,
    baseline_model_dir: Path | str | None = None,
    gptzero_model_dir: Path | str | None = None,
    batch_size: int = 64,
    score_splits: tuple[str, ...] = ("test",),
) -> dict:
    data_dir = Path(data_dir)
    run_dir = ensure_dir(run_dir)
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
            split_path = _resolve_split_path(data_dir, split)
        except FileNotFoundError:
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
    predictions_dir: Path | str,
    output_dir: Path | str,
    target_fpr: float = 0.01,
) -> dict:
    sample_frames = []
    prediction_frames = []
    for split in VALID_SCORE_SPLITS:
        try:
            sample_path = _resolve_split_path(data_dir, split)
        except FileNotFoundError:
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
    return evaluate_predictions(samples, predictions, output_dir, target_fpr=target_fpr)


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
        predictions_dir=Path(scoring["predictions_dir"]),
        output_dir=Path(config.baseline_run_dir) / "metrics",
        target_fpr=config.target_fpr,
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
        run_dir=config.gptzero_run_dir,
        baseline_model_dir=None,
        gptzero_model_dir=config.gptzero_model_dir,
        batch_size=config.row_batch_size,
        score_splits=score_splits,
    )
    evaluation = evaluate_run(
        data_dir=config.data_dir,
        predictions_dir=Path(scoring["predictions_dir"]),
        output_dir=Path(config.gptzero_run_dir) / "metrics",
        target_fpr=config.target_fpr,
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
    "GPTZERO_MODEL_DIR",
    "GPTZERO_RUN_DIR",
    "REFERENCE_GPTZERO_METRICS_DIR",
    "project_paths",
    "compare_reference_summaries",
    "evaluate_run",
    "find_split_path",
    "load_hc3_manifest",
    "load_metrics",
    "load_reference_metrics",
    "rebuild_hc3_from_source",
    "run_baseline_reference",
    "run_gptzero_experiment",
    "score_models",
    "train_baselines",
    "train_gptzero",
]
