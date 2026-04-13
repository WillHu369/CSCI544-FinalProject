from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gpt_zero.batching import BatchSource, SampleBatchLoader
from gpt_zero.config import DEFAULT_GPTZERO_LM, DEFAULT_RANDOM_STATE
from gpt_zero.io_utils import dump_json, ensure_dir, read_table, reset_dir
from gpt_zero.schemas import GPTZERO_DIAGNOSTIC_COLUMNS, ID_TO_LABEL, LABEL_TO_ID, PREDICTION_COLUMNS
from gpt_zero.text_utils import split_sentences

FEATURE_CACHE_VERSION = 3
DEFAULT_PERPLEXITY_BATCH_SIZE = 32


def _should_report_progress(batch_index: int, total_batches: int) -> bool:
    if total_batches <= 0:
        return batch_index == 1
    return batch_index == 1 or batch_index == total_batches or batch_index % 10 == 0


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _print_progress(
    prefix: str,
    batch_index: int,
    total_batches: int,
    processed_rows: int,
    total_rows: int,
    start_time: float,
) -> None:
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    rate = processed_rows / elapsed
    remaining_rows = max(total_rows - processed_rows, 0)
    eta_seconds = remaining_rows / rate if rate > 0 else 0.0
    print(
        f"[{prefix}] batch {batch_index}/{total_batches} | processed {processed_rows}/{total_rows} samples "
        f"| rate={rate:.2f} samples/s | elapsed={_format_duration(elapsed)} | eta={_format_duration(eta_seconds)}"
    )


@dataclass
class ScorerConfig:
    model_name: str = DEFAULT_GPTZERO_LM
    device: str = "auto"
    stride: int = 512
    max_length: int | None = None
    local_files_only: bool = False
    perplexity_batch_size: int = DEFAULT_PERPLEXITY_BATCH_SIZE


@dataclass
class FeatureExtractionConfig:
    max_sentences_per_text: int | None = None


class CausalLMPerplexityScorer:
    def __init__(self, config: ScorerConfig | None = None):
        self.config = config or ScorerConfig()
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._loss_fct = None

    def _resolve_device(self) -> str:
        assert self._torch is not None
        requested = (self.config.device or "auto").lower()
        if requested != "auto":
            if requested.startswith("cuda") and not self._torch.cuda.is_available():
                raise RuntimeError(
                    "Requested device='cuda', but the installed PyTorch build does not have CUDA support. "
                    "Rerun with '--device cpu' or install a CUDA-enabled PyTorch build in the virtual environment."
                )
            return self.config.device
        return "cuda" if self._torch.cuda.is_available() else "cpu"

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None and self._loss_fct is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The GPTZero-style detector requires 'torch' and 'transformers'. Install the project requirements first."
            ) from exc

        if self.config.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

        self._torch = torch
        load_kwargs = {"local_files_only": self.config.local_files_only}
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, **load_kwargs)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_load_kwargs = {**load_kwargs, "use_safetensors": False}
        self._model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **model_load_kwargs)
        self._model.eval()
        self._model.to(self._resolve_device())
        self._loss_fct = self._torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

        if self.config.max_length is None:
            self.config.max_length = int(
                getattr(self._model.config, "n_positions", getattr(self._model.config, "max_position_embeddings", 1024))
            )

    def resolved_feature_signature(self) -> dict[str, int | str]:
        if self.config.max_length is None:
            self._load()
        return {
            "model_name": self.config.model_name,
            "stride": int(self.config.stride),
            "max_length": int(self.config.max_length or 1024),
        }

    def perplexity(self, text: str) -> float:
        return self.perplexity_many([text], batch_size=1)[0]

    def perplexity_many(self, texts: list[str], batch_size: int | None = None) -> list[float]:
        self._load()
        assert self._torch is not None

        effective_batch_size = int(batch_size or self.config.perplexity_batch_size or DEFAULT_PERPLEXITY_BATCH_SIZE)
        if effective_batch_size <= 0:
            raise ValueError("perplexity batch size must be a positive integer.")

        cleaned_texts = [(text or "").strip() for text in texts]
        results = [float("nan")] * len(cleaned_texts)
        indexed_texts = [(index, text) for index, text in enumerate(cleaned_texts) if text]
        if not indexed_texts:
            return results

        indexed_texts.sort(key=lambda item: len(item[1]))
        for start in range(0, len(indexed_texts), effective_batch_size):
            chunk = indexed_texts[start : start + effective_batch_size]
            chunk_indices = [item[0] for item in chunk]
            chunk_texts = [item[1] for item in chunk]
            chunk_values = self._perplexity_many_chunk(chunk_texts)
            for original_index, value in zip(chunk_indices, chunk_values, strict=False):
                results[original_index] = value
        return results

    def _perplexity_many_chunk(self, texts: list[str]) -> list[float]:
        self._load()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._loss_fct is not None

        if not texts:
            return []

        encodings = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=False, verbose=False)
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        lengths = attention_mask.sum(dim=1, dtype=self._torch.long)
        num_rows = int(lengths.size(0))
        if num_rows == 0:
            return []

        max_length = int(self.config.max_length or 1024)
        stride = min(int(self.config.stride), max_length)
        device = self._resolve_device()

        nll_sums = self._torch.zeros(num_rows, dtype=self._torch.float64)
        token_counts = self._torch.zeros(num_rows, dtype=self._torch.long)
        previous_ends = self._torch.zeros(num_rows, dtype=self._torch.long)
        max_seq_len = int(lengths.max().item()) if lengths.numel() else 0
        valid = lengths >= 2

        for begin in range(0, max_seq_len, stride):
            active = (lengths > begin) & valid
            if not bool(active.any()):
                break

            active_indices = active.nonzero(as_tuple=False).squeeze(1)
            active_lengths = lengths[active_indices]
            window_end = self._torch.minimum(active_lengths, self._torch.full_like(active_lengths, begin + max_length))
            target_lengths = window_end - previous_ends[active_indices]
            keep = target_lengths > 0
            if not bool(keep.any()):
                continue

            active_indices = active_indices[keep]
            active_lengths = active_lengths[keep]
            window_end = window_end[keep]
            target_lengths = target_lengths[keep]
            if active_indices.numel() == 0:
                continue

            slice_end = int(window_end.max().item())
            input_slice = input_ids[active_indices, begin:slice_end]
            attention_slice = attention_mask[active_indices, begin:slice_end]
            target_ids = self._torch.full_like(input_slice, fill_value=-100)
            local_lengths = window_end - begin

            for row_index in range(input_slice.size(0)):
                local_length = int(local_lengths[row_index].item())
                target_length = int(target_lengths[row_index].item())
                start_index = max(local_length - target_length, 0)
                target_ids[row_index, start_index:local_length] = input_slice[row_index, start_index:local_length]

            with self._torch.no_grad():
                outputs = self._model(
                    input_ids=input_slice.to(device),
                    attention_mask=attention_slice.to(device),
                )
                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = target_ids[:, 1:].contiguous().to(device)
                token_losses = self._loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(shift_logits.size(0), -1)

            row_token_counts = (shift_labels != -100).sum(dim=1).cpu()
            row_loss_sums = token_losses.sum(dim=1).cpu().double()
            nll_sums[active_indices] += row_loss_sums
            token_counts[active_indices] += row_token_counts
            previous_ends[active_indices] = window_end

        values: list[float] = []
        for row_index in range(num_rows):
            token_count = int(token_counts[row_index].item())
            if token_count <= 0:
                values.append(float("nan"))
                continue
            mean_nll = nll_sums[row_index] / token_count
            values.append(float(self._torch.exp(mean_nll).item()))
        return values


def compute_burstiness(sentence_perplexities: list[float]) -> float:
    values = np.asarray([value for value in sentence_perplexities if not math.isnan(value)], dtype=float)
    if values.size < 2:
        return 0.0
    mean = float(values.mean())
    if mean <= 0:
        return 0.0
    return float(values.std(ddof=0)) / mean


def _sanitize_perplexity(value: float, fallback: float = 1.0) -> float:
    if math.isnan(value) or math.isinf(value) or value <= 0:
        return fallback
    return float(value)


def _score_texts(texts: list[str], scorer: Any, batch_size: int) -> list[float]:
    if not texts:
        return []
    if hasattr(scorer, "perplexity_many"):
        return [float(value) for value in scorer.perplexity_many(texts, batch_size=batch_size)]
    return [float(scorer.perplexity(text)) for text in texts]


def _capped_sentences(text: str, max_sentences_per_text: int | None) -> list[str]:
    sentences = [sentence for sentence in split_sentences(text) if sentence.strip()]
    if max_sentences_per_text is None:
        return sentences
    return sentences[: max(int(max_sentences_per_text), 0)]


def extract_batch_features(
    texts: list[str],
    scorer: Any,
    *,
    max_sentences_per_text: int | None = None,
    perplexity_batch_size: int = DEFAULT_PERPLEXITY_BATCH_SIZE,
) -> list[dict[str, float]]:
    cleaned_texts = [(text or "").strip() for text in texts]
    document_perplexities = [
        _sanitize_perplexity(value)
        for value in _score_texts(cleaned_texts, scorer, batch_size=perplexity_batch_size)
    ]
    sentence_lists = [_capped_sentences(text, max_sentences_per_text) for text in cleaned_texts]
    flattened_sentences = [sentence for sentences in sentence_lists for sentence in sentences]
    sentence_perplexities = [
        _sanitize_perplexity(value)
        for value in _score_texts(flattened_sentences, scorer, batch_size=perplexity_batch_size)
    ]

    features: list[dict[str, float]] = []
    cursor = 0
    for document_perplexity, sentences in zip(document_perplexities, sentence_lists, strict=False):
        count = len(sentences)
        current_sentence_values = sentence_perplexities[cursor : cursor + count]
        cursor += count
        valid_sentence_values = current_sentence_values or [document_perplexity]
        features.append(
            {
                "doc_perplexity": document_perplexity,
                "sentence_perplexity_mean": float(np.mean(valid_sentence_values)),
                "sentence_perplexity_std": float(np.std(valid_sentence_values, ddof=0)) if len(valid_sentence_values) > 1 else 0.0,
                "burstiness": compute_burstiness(valid_sentence_values),
            }
        )
    return features


def extract_text_features(text: str, scorer: Any) -> dict[str, float]:
    return extract_batch_features([text], scorer, perplexity_batch_size=1)[0]


def _feature_cache_paths(
    cache_dir: Path | str,
    source: BatchSource,
    scorer_config: ScorerConfig,
    feature_config: FeatureExtractionConfig,
) -> tuple[Path, Path, dict[str, Any]] | None:
    if isinstance(source, pd.DataFrame):
        return None

    source_path = Path(source)
    if not source_path.exists():
        return None

    cache_root = ensure_dir(cache_dir)
    source_stat = source_path.stat()
    cache_spec = {
        "version": FEATURE_CACHE_VERSION,
        "source_path": str(source_path.resolve()),
        "source_size": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "scorer": {
            "model_name": scorer_config.model_name,
            "stride": int(scorer_config.stride),
            "max_length": int(scorer_config.max_length or 1024),
        },
        "feature_config": asdict(feature_config),
        "feature_columns": list(GPTZERO_DIAGNOSTIC_COLUMNS),
    }
    digest = hashlib.sha1(json.dumps(cache_spec, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    base_name = f"{source_path.stem}__{digest}"
    return cache_root / f"{base_name}.parquet", cache_root / f"{base_name}.json", cache_spec


class GPTZeroLikeDetector:
    feature_columns = GPTZERO_DIAGNOSTIC_COLUMNS

    def __init__(
        self,
        scorer_config: ScorerConfig | None = None,
        feature_config: FeatureExtractionConfig | None = None,
        calibrator: Pipeline | None = None,
    ):
        self.scorer_config = scorer_config or ScorerConfig()
        self.feature_config = feature_config or FeatureExtractionConfig()
        self.calibrator = calibrator

    def build_feature_frame(
        self,
        samples: BatchSource,
        scorer: Any | None = None,
        batch_size: int = 32,
        progress_label: str = "gptzero",
        cache_dir: Path | str | None = None,
    ) -> pd.DataFrame:
        scorer = scorer or CausalLMPerplexityScorer(self.scorer_config)
        if isinstance(scorer, CausalLMPerplexityScorer) and self.scorer_config.max_length is None:
            scorer.resolved_feature_signature()

        cache_entry = None
        if cache_dir is not None:
            cache_entry = _feature_cache_paths(cache_dir, samples, self.scorer_config, self.feature_config)
            if cache_entry is not None:
                cache_frame_path, cache_meta_path, _ = cache_entry
                if cache_frame_path.exists():
                    print(f"[{progress_label}] loading cached GPTZero-style features from {cache_frame_path}")
                    try:
                        return read_table(cache_frame_path)
                    except Exception as exc:
                        print(f"[{progress_label}] cache read failed ({exc}); recomputing features")
                        cache_frame_path.unlink(missing_ok=True)
                        cache_meta_path.unlink(missing_ok=True)

        loader = SampleBatchLoader.from_source(samples, batch_size=batch_size, columns=("sample_id", "text", "label"))
        total_rows = loader.num_rows()
        total_batches = loader.num_batches()
        processed_rows = 0
        feature_batches = []
        start_time = time.perf_counter()
        print(f"[{progress_label}] extracting GPTZero-style features for {total_rows} samples in {total_batches} batches")
        for batch_index, batch in enumerate(loader.iter_frames(), start=1):
            texts = batch["text"].fillna("").astype(str).tolist()
            feature_frame = pd.DataFrame(
                extract_batch_features(
                    texts,
                    scorer,
                    max_sentences_per_text=self.feature_config.max_sentences_per_text,
                    perplexity_batch_size=max(1, int(self.scorer_config.perplexity_batch_size)),
                )
            )
            feature_frame.insert(0, "sample_id", batch["sample_id"].tolist())
            if "label" in batch.columns:
                feature_frame["label"] = batch["label"].tolist()
            feature_batches.append(feature_frame)
            processed_rows += len(batch)
            if _should_report_progress(batch_index, total_batches):
                _print_progress(progress_label, batch_index, total_batches, processed_rows, total_rows, start_time)

        elapsed = time.perf_counter() - start_time
        print(
            f"[{progress_label}] completed feature extraction for {processed_rows} samples "
            f"in {_format_duration(elapsed)}"
        )

        if not feature_batches:
            columns = ["sample_id", *self.feature_columns, "label"]
            feature_frame = pd.DataFrame(columns=columns)
        else:
            feature_frame = pd.concat(feature_batches, ignore_index=True)

        if cache_entry is not None:
            cache_frame_path, cache_meta_path, cache_spec = cache_entry
            temporary_cache = cache_frame_path.parent / f".{cache_frame_path.name}.{uuid4().hex}.tmp.parquet"
            feature_frame.to_parquet(temporary_cache, index=False)
            temporary_cache.replace(cache_frame_path)
            dump_json(
                {
                    **cache_spec,
                    "num_rows": int(len(feature_frame)),
                    "created_by": "gptzero_like",
                },
                cache_meta_path,
            )
            print(f"[{progress_label}] cached GPTZero-style features at {cache_frame_path}")

        return feature_frame

    def _to_model_matrix(self, feature_frame: pd.DataFrame) -> np.ndarray:
        base = feature_frame[self.feature_columns].astype(float).copy()
        base["doc_perplexity"] = np.log1p(base["doc_perplexity"])
        base["sentence_perplexity_mean"] = np.log1p(base["sentence_perplexity_mean"])
        return base.to_numpy(dtype=float)

    def fit_from_features(self, feature_frame: pd.DataFrame, labels: pd.Series) -> "GPTZeroLikeDetector":
        x = self._to_model_matrix(feature_frame)
        y = labels.map(LABEL_TO_ID).to_numpy(dtype=int)
        self.calibrator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=1000, random_state=DEFAULT_RANDOM_STATE, class_weight="balanced")),
            ]
        )
        self.calibrator.fit(x, y)
        return self

    def predict_from_features(self, feature_frame: pd.DataFrame, run_id: str) -> pd.DataFrame:
        if self.calibrator is None:
            raise RuntimeError("The GPTZero-like detector has not been fitted yet.")
        x = self._to_model_matrix(feature_frame)
        probabilities = self.calibrator.predict_proba(x)[:, 1]
        if hasattr(self.calibrator, "decision_function"):
            scores = self.calibrator.decision_function(x)
        else:
            clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
            scores = np.log(clipped / (1 - clipped))
        labels = [ID_TO_LABEL[int(value)] for value in (probabilities >= 0.5).astype(int)]
        predictions = pd.DataFrame(
            {
                "run_id": run_id,
                "detector_name": "gptzero_like",
                "sample_id": feature_frame["sample_id"].tolist(),
                "score": scores,
                "prob_ai": probabilities,
                "pred_label": labels,
            }
        )
        return pd.concat([predictions[PREDICTION_COLUMNS], feature_frame[self.feature_columns].reset_index(drop=True)], axis=1)

    def save(self, output_dir: Path | str) -> None:
        destination = ensure_dir(output_dir)
        if self.calibrator is None:
            raise RuntimeError("Cannot save an unfitted GPTZero-like detector.")
        temporary_joblib = Path(destination) / f".gptzero_like.{uuid4().hex}.tmp.joblib"
        joblib.dump(
            {
                "scorer_config": asdict(self.scorer_config),
                "feature_config": asdict(self.feature_config),
                "calibrator": self.calibrator,
            },
            temporary_joblib,
        )
        temporary_joblib.replace(Path(destination) / "gptzero_like.joblib")
        dump_json(
            {
                "detector_name": "gptzero_like",
                "feature_columns": self.feature_columns,
                "scorer_config": asdict(self.scorer_config),
                "feature_config": asdict(self.feature_config),
            },
            Path(destination) / "metadata.json",
        )

    @classmethod
    def load(cls, output_dir: Path | str) -> "GPTZeroLikeDetector":
        payload = joblib.load(Path(output_dir) / "gptzero_like.joblib")
        scorer_config = ScorerConfig(**payload["scorer_config"])
        feature_config = FeatureExtractionConfig(**payload.get("feature_config", {}))
        return cls(scorer_config=scorer_config, feature_config=feature_config, calibrator=payload["calibrator"])


def train_gptzero_like_detector(
    train_source: BatchSource,
    val_source: BatchSource | None,
    model_dir: Path | str,
    scorer_config: ScorerConfig,
    feature_config: FeatureExtractionConfig | None = None,
    batch_size: int = 32,
) -> dict:
    destination = reset_dir(model_dir, preserve_names=(".gitkeep",))
    detector = GPTZeroLikeDetector(scorer_config=scorer_config, feature_config=feature_config)
    scorer = CausalLMPerplexityScorer(scorer_config)
    cache_dir = ensure_dir(Path(destination) / "feature_cache")

    print("[gptzero] starting feature extraction for train split")
    train_features = detector.build_feature_frame(
        train_source,
        scorer=scorer,
        batch_size=batch_size,
        progress_label="gptzero-train",
        cache_dir=cache_dir,
    )

    calibration_split = "train"
    calibration_features = train_features
    if val_source is not None:
        calibration_split = "val"
        print(
            f"[gptzero] train feature extraction complete | rows={len(train_features)} "
            f"| moving to validation calibration split"
        )
        calibration_features = detector.build_feature_frame(
            val_source,
            scorer=scorer,
            batch_size=batch_size,
            progress_label="gptzero-calibration",
            cache_dir=cache_dir,
        )
        print(f"[gptzero] calibration feature extraction complete | rows={len(calibration_features)}")
    else:
        print(f"[gptzero] train feature extraction complete | rows={len(train_features)} | reusing train split as holdout")

    print("[gptzero] fitting detector on train features")
    detector.fit_from_features(train_features, train_features["label"])
    detector.scorer_config.local_files_only = True
    detector.save(destination)
    print("[gptzero] saved detector artifacts")

    metadata = {
        "detector_name": "gptzero_like",
        "fit_split": "train",
        "calibration_split": calibration_split,
        "num_train_samples": int(len(train_features)),
        "num_calibration_samples": int(len(calibration_features)),
        "feature_cache_dir": str(cache_dir),
        "scorer_config": asdict(scorer_config),
        "feature_config": asdict(detector.feature_config),
        "row_batch_size": batch_size,
        "perplexity_batch_size": int(scorer_config.perplexity_batch_size),
    }
    dump_json(metadata, Path(destination) / "training_summary.json")
    return metadata
