from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gpt_zero.classical import BaselineTrainingConfig, ClassicalBaselineSuite, train_classical_baselines
from gpt_zero.config import DATA_DIR, DEFAULT_GPTZERO_LM, DEFAULT_RANDOM_STATE, MODELS_DIR, RUNS_DIR, ensure_default_directories
from gpt_zero.gptzero_like import (
    CausalLMPerplexityScorer,
    FeatureExtractionConfig,
    GPTZeroLikeDetector,
    ScorerConfig,
    train_gptzero_like_detector,
)
from gpt_zero.hc3 import PrepareHC3Config, prepare_hc3_dataset, sample_prepared_dataset
from gpt_zero.io_utils import dump_json, ensure_dir, read_table, timestamp_run_id, write_table
from gpt_zero.metrics import evaluate_predictions
from gpt_zero.tfidf import TfidfFeatureConfig

VALID_SCORE_SPLITS = ("train", "val", "test")


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


def command_prepare_hc3(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    config = PrepareHC3Config(
        output_dir=output_dir,
        input_file=Path(args.input_file) if args.input_file else None,
        hf_dataset=args.hf_dataset,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        max_samples_per_label=args.max_samples_per_label,
        group_by_prompt=not args.row_split,
    )
    manifest = prepare_hc3_dataset(config)
    print(f"Prepared HC3 dataset at {output_dir}")
    print(manifest)


def command_prepare_hc3_fraction(args: argparse.Namespace) -> None:
    manifest = sample_prepared_dataset(
        input_dir=Path(args.input_data_dir),
        output_dir=Path(args.output_dir),
        fraction=args.fraction,
        random_state=args.random_state,
    )
    print(f"Prepared HC3 fraction dataset at {args.output_dir}")
    print(manifest)


def command_train_gptzero_like(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    train_source = data_dir / "train.parquet"
    val_source = (data_dir / "val.parquet") if (data_dir / "val.parquet").exists() else None

    print(f"[cli] training GPTZero-style detector from {train_source}")
    scorer_config = ScorerConfig(
        model_name=args.lm_model,
        device=args.device,
        stride=args.stride,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
        perplexity_batch_size=args.perplexity_batch_size,
    )
    feature_config = FeatureExtractionConfig(max_sentences_per_text=args.max_sentences_per_text)
    metadata = train_gptzero_like_detector(
        train_source,
        val_source,
        model_dir,
        scorer_config,
        feature_config=feature_config,
        batch_size=args.batch_size,
    )
    print(f"Saved GPTZero-like detector to {model_dir}")
    print(metadata)


def command_train_baselines(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    train_source = data_dir / "train.parquet"
    val_source = (data_dir / "val.parquet") if (data_dir / "val.parquet").exists() else None

    print(f"[cli] training baseline models from {train_source}")
    feature_config = TfidfFeatureConfig(
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
        min_df=args.min_df,
    )
    training_config = BaselineTrainingConfig(
        random_state=args.random_state,
        svm_c=args.svm_c,
        batch_size=args.batch_size,
        xgb_batch_size=args.xgb_batch_size,
        xgb_estimators=args.xgb_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
        xgb_device=args.xgb_device,
        xgb_early_stopping_rounds=args.xgb_early_stopping_rounds,
        xgb_eval_log_interval=args.xgb_eval_log_interval,
    )
    metadata = train_classical_baselines(
        train_source,
        model_dir,
        feature_config=feature_config,
        training_config=training_config,
        val_source=val_source,
    )
    print(f"Saved baseline models to {model_dir}")
    print(metadata)


def command_score_all(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    requested_splits = _normalize_score_splits(args.score_splits)
    run_id = args.run_id or timestamp_run_id("run")
    run_dir = ensure_dir(Path(args.output_dir) if args.output_dir else RUNS_DIR / run_id)
    predictions_dir = ensure_dir(run_dir / "predictions")

    gptzero_detector = None
    gptzero_scorer = None
    gptzero_cache_dir = None
    if args.gptzero_model_dir:
        gptzero_model_dir = Path(args.gptzero_model_dir)
        gptzero_model_file = gptzero_model_dir / "gptzero_like.joblib"
        if gptzero_model_file.exists():
            gptzero_detector = GPTZeroLikeDetector.load(gptzero_model_dir)
            gptzero_scorer = CausalLMPerplexityScorer(gptzero_detector.scorer_config)
            gptzero_cache_dir = gptzero_model_dir / "feature_cache"
    else:
        gptzero_model_dir = None

    baselines_suite = None
    if args.baselines_dir and Path(args.baselines_dir).exists():
        baselines_suite = ClassicalBaselineSuite.load(args.baselines_dir)

    if gptzero_detector is None and baselines_suite is None:
        raise RuntimeError(
            "No trained detectors were found. If GPTZero-style training failed, rerun it successfully "
            "or point '--gptzero-model-dir' to a directory containing gptzero_like.joblib."
        )

    for split in requested_splits:
        split_path = data_dir / f"{split}.parquet"
        if not split_path.exists():
            continue
        print(f"[cli] scoring split '{split}' from {split_path}")
        split_predictions = []
        if gptzero_detector is not None:
            feature_frame = gptzero_detector.build_feature_frame(
                split_path,
                scorer=gptzero_scorer,
                batch_size=args.batch_size,
                progress_label=f"gptzero-{split}",
                cache_dir=gptzero_cache_dir,
            )
            split_predictions.append(gptzero_detector.predict_from_features(feature_frame, run_id))
        if baselines_suite is not None:
            split_predictions.append(baselines_suite.predict(split_path, run_id, batch_size=args.batch_size))

        combined = pd.concat(split_predictions, ignore_index=True)
        write_table(combined, predictions_dir / f"{split}.parquet")

    dump_json(
        {
            "run_id": run_id,
            "data_dir": str(data_dir),
            "gptzero_model_dir": str(gptzero_model_dir) if gptzero_model_dir is not None else None,
            "baselines_dir": args.baselines_dir,
            "score_splits": requested_splits,
        },
        run_dir / "run_config.json",
    )
    print(f"Saved predictions to {predictions_dir}")


def command_evaluate(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)

    sample_frames = []
    prediction_frames = []
    for split in VALID_SCORE_SPLITS:
        sample_path = data_dir / f"{split}.parquet"
        prediction_path = predictions_dir / f"{split}.parquet"
        if sample_path.exists():
            sample_frames.append(read_table(sample_path))
        if prediction_path.exists():
            prediction_frames.append(read_table(prediction_path))

    if not sample_frames:
        raise RuntimeError(f"No prepared dataset files found under {data_dir}")
    if not prediction_frames:
        raise RuntimeError(f"No prediction files found under {predictions_dir}")

    sample_frame = pd.concat(sample_frames, ignore_index=True)
    prediction_frame = pd.concat(prediction_frames, ignore_index=True)
    summary = evaluate_predictions(sample_frame, prediction_frame, output_dir, target_fpr=args.target_fpr)
    print(f"Saved evaluation outputs to {output_dir}")
    print(summary)


def _run_profile(args: argparse.Namespace, profile: str) -> None:
    if profile == "small":
        data_dir = Path(args.data_dir or DATA_DIR / "hc3_small")
        baselines_dir = Path(args.baselines_dir or MODELS_DIR / "baselines_small")
        gptzero_dir = Path(args.gptzero_model_dir or MODELS_DIR / "gptzero_like_small")
        run_dir = Path(args.output_dir or RUNS_DIR / "hc3_small_run")
        max_samples_per_label = args.max_samples_per_label if args.max_samples_per_label is not None else 500
    else:
        data_dir = Path(args.data_dir or DATA_DIR / "hc3")
        baselines_dir = Path(args.baselines_dir or MODELS_DIR / "baselines")
        gptzero_dir = Path(args.gptzero_model_dir or MODELS_DIR / "gptzero_like")
        run_dir = Path(args.output_dir or RUNS_DIR / "hc3_full_run")
        max_samples_per_label = None

    if not args.skip_prepare:
        prepare_args = argparse.Namespace(
            output_dir=str(data_dir),
            input_file=args.input_file,
            hf_dataset=args.hf_dataset,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            max_samples_per_label=max_samples_per_label,
            row_split=args.row_split,
        )
        command_prepare_hc3(prepare_args)

    if not args.skip_baselines:
        baseline_args = argparse.Namespace(
            data_dir=str(data_dir),
            model_dir=str(baselines_dir),
            random_state=args.random_state,
            batch_size=args.batch_size,
            xgb_batch_size=args.xgb_batch_size,
            word_max_features=args.word_max_features,
            char_max_features=args.char_max_features,
            min_df=args.min_df,
            svm_c=args.svm_c,
            xgb_estimators=args.xgb_estimators,
            xgb_max_depth=args.xgb_max_depth,
            xgb_learning_rate=args.xgb_learning_rate,
            xgb_subsample=args.xgb_subsample,
            xgb_colsample_bytree=args.xgb_colsample_bytree,
            xgb_device=args.xgb_device,
            xgb_early_stopping_rounds=args.xgb_early_stopping_rounds,
            xgb_eval_log_interval=args.xgb_eval_log_interval,
        )
        command_train_baselines(baseline_args)

    if not args.skip_gptzero:
        gptzero_args = argparse.Namespace(
            data_dir=str(data_dir),
            model_dir=str(gptzero_dir),
            lm_model=args.lm_model,
            device=args.device,
            stride=args.stride,
            max_length=args.max_length,
            local_files_only=args.local_files_only,
            batch_size=args.batch_size,
            perplexity_batch_size=args.perplexity_batch_size,
            max_sentences_per_text=args.max_sentences_per_text,
        )
        command_train_gptzero_like(gptzero_args)

    score_args = argparse.Namespace(
        data_dir=str(data_dir),
        gptzero_model_dir=str(gptzero_dir) if gptzero_dir is not None else None,
        baselines_dir=str(baselines_dir) if baselines_dir is not None else None,
        output_dir=str(run_dir),
        run_id=args.run_id,
        batch_size=args.batch_size,
        score_splits=args.score_splits,
    )
    command_score_all(score_args)

    evaluate_args = argparse.Namespace(
        data_dir=str(data_dir),
        predictions_dir=str(run_dir / "predictions"),
        output_dir=str(run_dir / "metrics"),
        target_fpr=args.target_fpr,
    )
    command_evaluate(evaluate_args)


def command_run_small(args: argparse.Namespace) -> None:
    _run_profile(args, profile="small")


def command_run_full(args: argparse.Namespace) -> None:
    _run_profile(args, profile="full")


def command_run_gptzero_fraction(args: argparse.Namespace) -> None:
    source_data_dir = Path(args.source_data_dir)
    data_dir = Path(args.data_dir or DATA_DIR / "hc3_gptzero_fraction")
    gptzero_dir = Path(args.gptzero_model_dir or MODELS_DIR / "gptzero_like_fraction")
    run_dir = Path(args.output_dir or RUNS_DIR / "hc3_gptzero_fraction_run")

    if not args.skip_prepare:
        fraction_args = argparse.Namespace(
            input_data_dir=str(source_data_dir),
            output_dir=str(data_dir),
            fraction=args.fraction,
            random_state=args.random_state,
        )
        command_prepare_hc3_fraction(fraction_args)

    gptzero_args = argparse.Namespace(
        data_dir=str(data_dir),
        model_dir=str(gptzero_dir),
        lm_model=args.lm_model,
        device=args.device,
        stride=args.stride,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
        batch_size=args.batch_size,
        perplexity_batch_size=args.perplexity_batch_size,
        max_sentences_per_text=args.max_sentences_per_text,
    )
    command_train_gptzero_like(gptzero_args)

    score_args = argparse.Namespace(
        data_dir=str(data_dir),
        gptzero_model_dir=str(gptzero_dir),
        baselines_dir=None,
        output_dir=str(run_dir),
        run_id=args.run_id,
        batch_size=args.batch_size,
        score_splits=args.score_splits,
    )
    command_score_all(score_args)

    evaluate_args = argparse.Namespace(
        data_dir=str(data_dir),
        predictions_dir=str(run_dir / "predictions"),
        output_dir=str(run_dir / "metrics"),
        target_fpr=args.target_fpr,
    )
    command_evaluate(evaluate_args)


def _add_shared_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--baselines-dir", default=None)
    parser.add_argument("--gptzero-model-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--hf-dataset", default="Hello-SimpleAI/HC3")
    parser.add_argument("--row-split", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--lm-model", default=DEFAULT_GPTZERO_LM)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--perplexity-batch-size", type=int, default=32)
    parser.add_argument("--max-sentences-per-text", type=int, default=None)
    parser.add_argument("--score-splits", nargs="+", default=list(VALID_SCORE_SPLITS))
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-gptzero", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--xgb-batch-size", type=int, default=1024)
    parser.add_argument("--word-max-features", type=int, default=20000)
    parser.add_argument("--char-max-features", type=int, default=15000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--xgb-estimators", type=int, default=120)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--xgb-device", default="cuda")
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=20)
    parser.add_argument("--xgb-eval-log-interval", type=int, default=10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local GPTZero-style detector and baseline training pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare_hc3", help="Download or normalize HC3 and emit train/val/test parquet splits.")
    prepare_parser.add_argument("--output-dir", default=str(DATA_DIR / "hc3"))
    prepare_parser.add_argument("--input-file", default=None)
    prepare_parser.add_argument("--hf-dataset", default="Hello-SimpleAI/HC3")
    prepare_parser.add_argument("--test-size", type=float, default=0.2)
    prepare_parser.add_argument("--val-size", type=float, default=0.1)
    prepare_parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    prepare_parser.add_argument("--max-samples-per-label", type=int, default=None)
    prepare_parser.add_argument("--row-split", action="store_true", help="Use the older row-level split instead of prompt-disjoint splitting.")
    prepare_parser.set_defaults(func=command_prepare_hc3)

    fraction_parser = subparsers.add_parser("prepare_hc3_fraction", help="Sample a prepared HC3 dataset by fraction while preserving split integrity.")
    fraction_parser.add_argument("--input-data-dir", default=str(DATA_DIR / "hc3"))
    fraction_parser.add_argument("--output-dir", default=str(DATA_DIR / "hc3_gptzero_fraction"))
    fraction_parser.add_argument("--fraction", type=float, default=0.1)
    fraction_parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    fraction_parser.set_defaults(func=command_prepare_hc3_fraction)

    gptzero_parser = subparsers.add_parser("train_gptzero_like", help="Train the GPTZero-style detector.")
    gptzero_parser.add_argument("--data-dir", default=str(DATA_DIR / "hc3"))
    gptzero_parser.add_argument("--model-dir", default=str(MODELS_DIR / "gptzero_like"))
    gptzero_parser.add_argument("--lm-model", default=DEFAULT_GPTZERO_LM)
    gptzero_parser.add_argument("--device", default="auto")
    gptzero_parser.add_argument("--stride", type=int, default=512)
    gptzero_parser.add_argument("--max-length", type=int, default=None)
    gptzero_parser.add_argument("--local-files-only", action="store_true")
    gptzero_parser.add_argument("--batch-size", type=int, default=32)
    gptzero_parser.add_argument("--perplexity-batch-size", type=int, default=32)
    gptzero_parser.add_argument("--max-sentences-per-text", type=int, default=None)
    gptzero_parser.set_defaults(func=command_train_gptzero_like)

    baseline_parser = subparsers.add_parser("train_baselines", help="Train TF-IDF SVM and XGBoost baselines.")
    baseline_parser.add_argument("--data-dir", default=str(DATA_DIR / "hc3"))
    baseline_parser.add_argument("--model-dir", default=str(MODELS_DIR / "baselines"))
    baseline_parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    baseline_parser.add_argument("--batch-size", type=int, default=2048)
    baseline_parser.add_argument("--xgb-batch-size", type=int, default=1024)
    baseline_parser.add_argument("--word-max-features", type=int, default=20000)
    baseline_parser.add_argument("--char-max-features", type=int, default=15000)
    baseline_parser.add_argument("--min-df", type=int, default=2)
    baseline_parser.add_argument("--svm-c", type=float, default=1.0)
    baseline_parser.add_argument("--xgb-estimators", type=int, default=120)
    baseline_parser.add_argument("--xgb-max-depth", type=int, default=4)
    baseline_parser.add_argument("--xgb-learning-rate", type=float, default=0.1)
    baseline_parser.add_argument("--xgb-subsample", type=float, default=0.8)
    baseline_parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    baseline_parser.add_argument("--xgb-device", default="cuda")
    baseline_parser.add_argument("--xgb-early-stopping-rounds", type=int, default=20)
    baseline_parser.add_argument("--xgb-eval-log-interval", type=int, default=10)
    baseline_parser.set_defaults(func=command_train_baselines)

    score_parser = subparsers.add_parser("score_all", help="Score any selected detectors on the requested splits.")
    score_parser.add_argument("--data-dir", default=str(DATA_DIR / "hc3"))
    score_parser.add_argument("--gptzero-model-dir", default=None)
    score_parser.add_argument("--baselines-dir", default=None)
    score_parser.add_argument("--output-dir", default=None)
    score_parser.add_argument("--run-id", default=None)
    score_parser.add_argument("--batch-size", type=int, default=1024)
    score_parser.add_argument("--score-splits", nargs="+", default=list(VALID_SCORE_SPLITS))
    score_parser.set_defaults(func=command_score_all)

    evaluate_parser = subparsers.add_parser("evaluate", help="Compute metrics, slice metrics, and ROC points.")
    evaluate_parser.add_argument("--data-dir", default=str(DATA_DIR / "hc3"))
    evaluate_parser.add_argument("--predictions-dir", required=True)
    evaluate_parser.add_argument("--output-dir", required=True)
    evaluate_parser.add_argument("--target-fpr", type=float, default=0.01)
    evaluate_parser.set_defaults(func=command_evaluate)

    run_small_parser = subparsers.add_parser("run_small", help="Run prepare, train, score, and evaluate on a smaller HC3 subset.")
    _add_shared_profile_args(run_small_parser)
    run_small_parser.add_argument("--max-samples-per-label", type=int, default=500)
    run_small_parser.set_defaults(func=command_run_small)

    run_full_parser = subparsers.add_parser("run_full", help="Run the full HC3 workflow end to end.")
    _add_shared_profile_args(run_full_parser)
    run_full_parser.set_defaults(func=command_run_full)

    run_fraction_parser = subparsers.add_parser("run_gptzero_fraction", help="Sample a fraction of prepared HC3 and run the GPTZero-style pipeline only.")
    run_fraction_parser.add_argument("--source-data-dir", default=str(DATA_DIR / "hc3"))
    run_fraction_parser.add_argument("--data-dir", default=str(DATA_DIR / "hc3_gptzero_fraction"))
    run_fraction_parser.add_argument("--gptzero-model-dir", default=str(MODELS_DIR / "gptzero_like_fraction"))
    run_fraction_parser.add_argument("--output-dir", default=str(RUNS_DIR / "hc3_gptzero_fraction_run"))
    run_fraction_parser.add_argument("--run-id", default=None)
    run_fraction_parser.add_argument("--fraction", type=float, default=0.1)
    run_fraction_parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    run_fraction_parser.add_argument("--lm-model", default=DEFAULT_GPTZERO_LM)
    run_fraction_parser.add_argument("--device", default="auto")
    run_fraction_parser.add_argument("--stride", type=int, default=512)
    run_fraction_parser.add_argument("--max-length", type=int, default=None)
    run_fraction_parser.add_argument("--local-files-only", action="store_true")
    run_fraction_parser.add_argument("--batch-size", type=int, default=128)
    run_fraction_parser.add_argument("--perplexity-batch-size", type=int, default=32)
    run_fraction_parser.add_argument("--max-sentences-per-text", type=int, default=None)
    run_fraction_parser.add_argument("--score-splits", nargs="+", default=["test"])
    run_fraction_parser.add_argument("--target-fpr", type=float, default=0.01)
    run_fraction_parser.add_argument("--skip-prepare", action="store_true")
    run_fraction_parser.set_defaults(func=command_run_gptzero_fraction)
    return parser


def main() -> None:
    ensure_default_directories()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
