from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import vstack
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from gpt_zero.batching import BatchSource, SampleBatchLoader
from gpt_zero.config import DEFAULT_RANDOM_STATE
from gpt_zero.io_utils import dump_json, ensure_dir, load_json, reset_dir
from gpt_zero.schemas import CLASSICAL_DIAGNOSTIC_COLUMNS, ID_TO_LABEL, LABEL_TO_ID, PREDICTION_COLUMNS
from gpt_zero.tfidf import TfidfFeatureConfig, TfidfFeatureExtractor


def _should_report_progress(batch_index: int, total_batches: int) -> bool:
    if total_batches <= 0:
        return batch_index == 1
    return batch_index == 1 or batch_index == total_batches or batch_index % 10 == 0


def _print_progress(prefix: str, batch_index: int, total_batches: int, processed_rows: int, total_rows: int) -> None:
    print(f"[{prefix}] batch {batch_index}/{total_batches} | processed {processed_rows}/{total_rows} samples")

class _XGBoostRoundProgressCallback(xgb.callback.TrainingCallback):
    def __init__(self, total_rounds: int, log_interval: int = 10):
        self.total_rounds = max(int(total_rounds), 1)
        self.log_interval = max(int(log_interval), 1)
        self.best_val_logloss: float | None = None
        self.best_round: int | None = None

    def _latest_metric(self, evals_log, dataset_name: str, metric_name: str) -> float | None:
        dataset_metrics = evals_log.get(dataset_name, {})
        metric_history = dataset_metrics.get(metric_name, [])
        if not metric_history:
            return None
        latest = metric_history[-1]
        if isinstance(latest, tuple):
            latest = latest[0]
        return float(latest)

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        round_number = epoch + 1
        val_logloss = self._latest_metric(evals_log, "val", "logloss")
        if val_logloss is not None and (self.best_val_logloss is None or val_logloss < self.best_val_logloss):
            self.best_val_logloss = val_logloss
            self.best_round = round_number

        should_print = round_number == 1 or round_number == self.total_rounds or round_number % self.log_interval == 0
        if should_print:
            print(f"[xgboost] round {round_number}/{self.total_rounds}")
        return False


@dataclass
class BaselineTrainingConfig:
    random_state: int = DEFAULT_RANDOM_STATE
    svm_c: float = 1.0
    svm_max_iter: int = 10000
    batch_size: int = 2048
    xgb_batch_size: int = 1024
    xgb_estimators: int = 120
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_max_bin: int = 256
    xgb_device: str = "cuda"
    xgb_early_stopping_rounds: int = 20
    xgb_eval_log_interval: int = 10


class _TfidfXGBoostDataIter(xgb.DataIter):
    def __init__(
        self,
        source: BatchSource,
        feature_extractor: TfidfFeatureExtractor,
        batch_size: int,
        progress_label: str,
        cache_prefix: str | None = None,
    ):
        super().__init__(cache_prefix=cache_prefix, release_data=True)
        self.source = source
        self.feature_extractor = feature_extractor
        self.batch_size = batch_size
        self.progress_label = progress_label
        self._iterator = None
        self._pass_index = 0
        loader = SampleBatchLoader.from_source(self.source, batch_size=self.batch_size, columns=("text", "label"))
        self._total_rows = loader.num_rows()
        self._total_batches = loader.num_batches()
        self._processed_rows = 0
        self.reset()

    def reset(self) -> None:
        self._pass_index += 1
        self._processed_rows = 0
        loader = SampleBatchLoader.from_source(self.source, batch_size=self.batch_size, columns=("text", "label"))
        self._iterator = enumerate(loader.iter_frames(), start=1)
        if self._pass_index <= 2 or self._pass_index % 10 == 0:
            print(
                f"[{self.progress_label}] starting streamed pass {self._pass_index} "
                f"over {self._total_rows} samples in {self._total_batches} batches"
            )

    def next(self, input_data: Any) -> bool:
        assert self._iterator is not None
        try:
            batch_index, frame = next(self._iterator)
        except StopIteration:
            return False

        x_batch = self.feature_extractor.transform(frame["text"].fillna("").astype(str).tolist())
        y_batch = np.asarray([LABEL_TO_ID[str(label)] for label in frame["label"]], dtype=np.float32)
        input_data(data=x_batch, label=y_batch)
        self._processed_rows += len(frame)
        if self._pass_index <= 2 and _should_report_progress(batch_index, self._total_batches):
            _print_progress(self.progress_label, batch_index, self._total_batches, self._processed_rows, self._total_rows)
        return True


class BatchedXGBoostBinaryClassifier:
    def __init__(self, training_config: BaselineTrainingConfig | None = None):
        self.training_config = training_config or BaselineTrainingConfig()
        self.booster: xgb.Booster | None = None

    def _params(self) -> dict[str, Any]:
        return {
            "max_depth": self.training_config.xgb_max_depth,
            "learning_rate": self.training_config.xgb_learning_rate,
            "subsample": self.training_config.xgb_subsample,
            "colsample_bytree": self.training_config.xgb_colsample_bytree,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": self.training_config.random_state,
            "seed": self.training_config.random_state,
            "tree_method": "hist",
            "max_bin": self.training_config.xgb_max_bin,
            "device": self.training_config.xgb_device,
            "nthread": -1,
        }

    def fit(
        self,
        train_source: BatchSource,
        feature_extractor: TfidfFeatureExtractor,
        val_source: BatchSource | None = None,
        cache_prefix: str | None = None,
    ) -> "BatchedXGBoostBinaryClassifier":
        train_iterator = _TfidfXGBoostDataIter(
            source=train_source,
            feature_extractor=feature_extractor,
            batch_size=self.training_config.xgb_batch_size,
            progress_label="xgboost-train-stream",
            cache_prefix=cache_prefix,
        )
        print("[xgboost] building training QuantileDMatrix")
        dtrain = xgb.QuantileDMatrix(train_iterator, max_bin=self.training_config.xgb_max_bin)
        print("[xgboost] training QuantileDMatrix ready")
        evals = [(dtrain, "train")]
        if val_source is not None:
            val_iterator = _TfidfXGBoostDataIter(
                source=val_source,
                feature_extractor=feature_extractor,
                batch_size=self.training_config.xgb_batch_size,
                progress_label="xgboost-val-stream",
            )
            print("[xgboost] building validation QuantileDMatrix")
            dval = xgb.QuantileDMatrix(val_iterator, max_bin=self.training_config.xgb_max_bin, ref=dtrain)
            print("[xgboost] validation QuantileDMatrix ready")
            evals.append((dval, "val"))
        print("[xgboost] entering boosting loop")
        self.booster = xgb.train(
            params=self._params(),
            dtrain=dtrain,
            num_boost_round=self.training_config.xgb_estimators,
            evals=evals,
            early_stopping_rounds=self.training_config.xgb_early_stopping_rounds if val_source is not None else None,
            verbose_eval=False,
            callbacks=[
                _XGBoostRoundProgressCallback(
                    self.training_config.xgb_estimators,
                    log_interval=self.training_config.xgb_eval_log_interval,
                )
            ],
        )
        boosted_rounds = self.booster.num_boosted_rounds()
        best_iteration = getattr(self.booster, "best_iteration", None)
        best_score = getattr(self.booster, "best_score", None)
        if best_iteration is not None and best_score is not None:
            print(
                f"[xgboost] best validation logloss {float(best_score):.4f} at round {int(best_iteration) + 1}; "
                f"finished after {boosted_rounds} boosting rounds"
            )
        else:
            print(f"[xgboost] finished after {boosted_rounds} boosting rounds")
        return self

    def predict_proba(self, x) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("The XGBoost baseline has not been fitted yet.")
        probabilities = np.asarray(self.booster.predict(xgb.DMatrix(x)), dtype=np.float32).reshape(-1)
        return np.column_stack([1.0 - probabilities, probabilities])

    def save(self, path: Path | str) -> None:
        if self.booster is None:
            raise RuntimeError("Cannot save an unfitted XGBoost baseline.")
        self.booster.save_model(Path(path))

    @classmethod
    def load(cls, path: Path | str, training_config: BaselineTrainingConfig | None = None) -> "BatchedXGBoostBinaryClassifier":
        model = cls(training_config=training_config)
        booster = xgb.Booster()
        booster.load_model(Path(path))
        model.booster = booster
        return model


class ClassicalBaselineSuite:
    def __init__(
        self,
        feature_config: TfidfFeatureConfig | None = None,
        training_config: BaselineTrainingConfig | None = None,
    ):
        self.feature_extractor = TfidfFeatureExtractor(feature_config)
        self.training_config = training_config or BaselineTrainingConfig()
        self.svm_model: LinearSVC | None = None
        self.svm_calibrator: LogisticRegression | None = None
        self.xgb_model: BatchedXGBoostBinaryClassifier | None = None

    def _collect_texts(self, source: BatchSource) -> list[str]:
        loader = SampleBatchLoader.from_source(source, batch_size=self.training_config.batch_size, columns=("text",))
        total_rows = loader.num_rows()
        total_batches = loader.num_batches()
        processed_rows = 0
        texts: list[str] = []
        print(f"[tfidf] fitting vectorizers on {total_rows} samples in {total_batches} batches")
        for batch_index, frame in enumerate(loader.iter_frames(), start=1):
            batch_texts = frame["text"].fillna("").astype(str).tolist()
            texts.extend(batch_texts)
            processed_rows += len(batch_texts)
            if _should_report_progress(batch_index, total_batches):
                _print_progress("tfidf", batch_index, total_batches, processed_rows, total_rows)
        return texts

    def _vectorize_source(self, source: BatchSource, progress_label: str):
        loader = SampleBatchLoader.from_source(source, batch_size=self.training_config.batch_size, columns=("text", "label"))
        total_rows = loader.num_rows()
        total_batches = loader.num_batches()
        processed_rows = 0
        matrices = []
        labels = []
        print(f"[{progress_label}] vectorizing {total_rows} samples in {total_batches} batches")
        for batch_index, frame in enumerate(loader.iter_frames(), start=1):
            matrices.append(self.feature_extractor.transform(frame["text"].fillna("").astype(str).tolist()))
            labels.append(np.asarray([LABEL_TO_ID[str(label)] for label in frame["label"]], dtype=int))
            processed_rows += len(frame)
            if _should_report_progress(batch_index, total_batches):
                _print_progress(progress_label, batch_index, total_batches, processed_rows, total_rows)
        if not matrices:
            raise ValueError("No samples were available for baseline training.")
        return vstack(matrices, format="csr"), np.concatenate(labels)

    def _fit_svm(self, train_source: BatchSource, calibration_source: BatchSource) -> None:
        x_train, y_train = self._vectorize_source(train_source, progress_label="svm-train")
        print(f"[svm] fitting LinearSVC on {len(y_train)} samples")
        self.svm_model = LinearSVC(
            C=self.training_config.svm_c,
            class_weight="balanced",
            random_state=self.training_config.random_state,
            max_iter=self.training_config.svm_max_iter,
        )
        self.svm_model.fit(x_train, y_train)
        print("[svm] LinearSVC fit complete")

        calibration_loader = SampleBatchLoader.from_source(
            calibration_source,
            batch_size=self.training_config.batch_size,
            columns=("text", "label"),
        )
        total_rows = calibration_loader.num_rows()
        total_batches = calibration_loader.num_batches()
        processed_rows = 0
        calibration_scores = []
        calibration_labels = []
        print(f"[svm] calibrating probabilities on {total_rows} samples in {total_batches} batches")
        for batch_index, frame in enumerate(calibration_loader.iter_frames(), start=1):
            x_batch = self.feature_extractor.transform(frame["text"].fillna("").astype(str).tolist())
            calibration_scores.append(self.svm_model.decision_function(x_batch))
            calibration_labels.append(np.asarray([LABEL_TO_ID[str(label)] for label in frame["label"]], dtype=int))
            processed_rows += len(frame)
            if _should_report_progress(batch_index, total_batches):
                _print_progress("svm-calibration", batch_index, total_batches, processed_rows, total_rows)

        self.svm_calibrator = LogisticRegression(max_iter=1000, random_state=self.training_config.random_state)
        self.svm_calibrator.fit(np.concatenate(calibration_scores).reshape(-1, 1), np.concatenate(calibration_labels))
        print("[svm] calibration fit complete")

        del x_train
        del y_train
        gc.collect()

    def fit(self, train_source: BatchSource, val_source: BatchSource | None = None) -> "ClassicalBaselineSuite":
        self.feature_extractor.fit(self._collect_texts(train_source))
        calibration_source = val_source if val_source is not None else train_source
        self._fit_svm(train_source, calibration_source)

        self.xgb_model = BatchedXGBoostBinaryClassifier(self.training_config)
        print("[xgboost] fitting streamed XGBoost baseline")
        self.xgb_model.fit(train_source, self.feature_extractor, val_source=val_source)
        print("[xgboost] training complete")
        return self

    def _build_prediction_frame(
        self,
        detector_name: str,
        sample_ids: list[str],
        probabilities: np.ndarray,
        margins: np.ndarray,
        run_id: str,
    ) -> pd.DataFrame:
        clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
        scores = np.log(clipped / (1 - clipped))
        labels = [ID_TO_LABEL[int(value)] for value in (probabilities >= 0.5).astype(int)]
        return pd.DataFrame(
            {
                "run_id": run_id,
                "detector_name": detector_name,
                "sample_id": sample_ids,
                "score": scores,
                "prob_ai": probabilities,
                "pred_label": labels,
                "margin": margins,
            }
        )[PREDICTION_COLUMNS + CLASSICAL_DIAGNOSTIC_COLUMNS]

    def _predict_svm_batches(self, samples: BatchSource, run_id: str, batch_size: int) -> list[pd.DataFrame]:
        if self.svm_model is None or self.svm_calibrator is None:
            raise RuntimeError("The SVM baseline has not been fitted yet.")
        frames = []
        loader = SampleBatchLoader.from_source(samples, batch_size=batch_size, columns=("sample_id", "text"))
        total_rows = loader.num_rows()
        total_batches = loader.num_batches()
        processed_rows = 0
        print(f"[svm] scoring {total_rows} samples in {total_batches} batches")
        for batch_index, frame in enumerate(loader.iter_frames(), start=1):
            x_batch = self.feature_extractor.transform(frame["text"].fillna("").astype(str).tolist())
            margins = np.asarray(self.svm_model.decision_function(x_batch), dtype=float).reshape(-1)
            probabilities = self.svm_calibrator.predict_proba(margins.reshape(-1, 1))[:, 1]
            frames.append(
                self._build_prediction_frame(
                    "svm_tfidf",
                    frame["sample_id"].tolist(),
                    probabilities,
                    margins,
                    run_id,
                )
            )
            processed_rows += len(frame)
            if _should_report_progress(batch_index, total_batches):
                _print_progress("svm-score", batch_index, total_batches, processed_rows, total_rows)
        return frames

    def _predict_xgb_batches(self, samples: BatchSource, run_id: str, batch_size: int) -> list[pd.DataFrame]:
        if self.xgb_model is None:
            raise RuntimeError("The XGBoost baseline has not been fitted yet.")
        frames = []
        loader = SampleBatchLoader.from_source(samples, batch_size=batch_size, columns=("sample_id", "text"))
        total_rows = loader.num_rows()
        total_batches = loader.num_batches()
        processed_rows = 0
        print(f"[xgboost] scoring {total_rows} samples in {total_batches} batches")
        for batch_index, frame in enumerate(loader.iter_frames(), start=1):
            x_batch = self.feature_extractor.transform(frame["text"].fillna("").astype(str).tolist())
            probabilities = self.xgb_model.predict_proba(x_batch)[:, 1]
            margins = np.log(np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1.0))
            frames.append(
                self._build_prediction_frame(
                    "xgboost_tfidf",
                    frame["sample_id"].tolist(),
                    probabilities,
                    margins,
                    run_id,
                )
            )
            processed_rows += len(frame)
            if _should_report_progress(batch_index, total_batches):
                _print_progress("xgboost-score", batch_index, total_batches, processed_rows, total_rows)
        return frames

    def predict(self, samples: BatchSource, run_id: str, batch_size: int | None = None) -> pd.DataFrame:
        prediction_batch_size = batch_size or self.training_config.batch_size
        frames = self._predict_svm_batches(samples, run_id, prediction_batch_size)
        frames.extend(self._predict_xgb_batches(samples, run_id, prediction_batch_size))
        return pd.concat(frames, ignore_index=True)

    def save(self, output_dir: Path | str) -> None:
        destination = ensure_dir(output_dir)
        self.feature_extractor.save(destination)
        if self.svm_model is None or self.svm_calibrator is None or self.xgb_model is None:
            raise RuntimeError("Cannot save an unfitted baseline suite.")
        joblib.dump(self.svm_model, Path(destination) / "svm_model.joblib")
        joblib.dump(self.svm_calibrator, Path(destination) / "svm_calibrator.joblib")
        self.xgb_model.save(Path(destination) / "xgboost.json")
        dump_json(
            {
                "detectors": ["svm_tfidf", "xgboost_tfidf"],
                "training_config": self.training_config.__dict__,
                "feature_config": self.feature_extractor.config.__dict__,
            },
            Path(destination) / "metadata.json",
        )

    @classmethod
    def load(cls, output_dir: Path | str) -> "ClassicalBaselineSuite":
        destination = Path(output_dir)
        metadata = load_json(destination / "metadata.json")
        suite = cls(
            feature_config=TfidfFeatureConfig(**metadata["feature_config"]),
            training_config=BaselineTrainingConfig(**metadata["training_config"]),
        )
        suite.feature_extractor = TfidfFeatureExtractor.load(destination)
        suite.svm_model = joblib.load(destination / "svm_model.joblib")
        suite.svm_calibrator = joblib.load(destination / "svm_calibrator.joblib")
        suite.xgb_model = BatchedXGBoostBinaryClassifier.load(destination / "xgboost.json", suite.training_config)
        return suite


def train_classical_baselines(
    train_source: BatchSource,
    model_dir: Path | str,
    feature_config: TfidfFeatureConfig | None = None,
    training_config: BaselineTrainingConfig | None = None,
    val_source: BatchSource | None = None,
) -> dict:
    destination = reset_dir(model_dir)
    suite = ClassicalBaselineSuite(feature_config=feature_config, training_config=training_config)
    suite.fit(train_source, val_source=val_source)
    suite.save(destination)

    if isinstance(train_source, pd.DataFrame):
        num_train_samples = len(train_source)
    else:
        loader = SampleBatchLoader.from_source(train_source, batch_size=suite.training_config.batch_size)
        num_train_samples = sum(len(frame) for frame in loader.iter_frames())

    metadata = {
        "detectors": ["svm_tfidf", "xgboost_tfidf"],
        "num_train_samples": int(num_train_samples),
        "feature_config": suite.feature_extractor.config.__dict__,
        "training_config": suite.training_config.__dict__,
        "calibration_split": "val" if val_source is not None else "train",
    }
    dump_json(metadata, Path(destination) / "training_summary.json")
    return metadata
