from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from gpt_zero.io_utils import dump_json, ensure_dir


@dataclass
class TfidfFeatureConfig:
    word_ngram_min: int = 1
    word_ngram_max: int = 2
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    word_max_features: int = 20000
    char_max_features: int = 15000
    min_df: int = 2


class TfidfFeatureExtractor:
    def __init__(self, config: TfidfFeatureConfig | None = None):
        self.config = config or TfidfFeatureConfig()
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            strip_accents="unicode",
            lowercase=True,
            sublinear_tf=True,
            ngram_range=(self.config.word_ngram_min, self.config.word_ngram_max),
            max_features=self.config.word_max_features,
            min_df=self.config.min_df,
            dtype=np.float32,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            strip_accents="unicode",
            lowercase=True,
            sublinear_tf=True,
            ngram_range=(self.config.char_ngram_min, self.config.char_ngram_max),
            max_features=self.config.char_max_features,
            min_df=self.config.min_df,
            dtype=np.float32,
        )

    def _materialize_texts(self, texts: Iterable[str]) -> list[str]:
        return [str(text or "") for text in texts]

    def fit(self, texts: Iterable[str]) -> "TfidfFeatureExtractor":
        materialized = self._materialize_texts(texts)
        self.word_vectorizer.fit(materialized)
        self.char_vectorizer.fit(materialized)
        return self

    def fit_transform(self, texts: Iterable[str]):
        materialized = self._materialize_texts(texts)
        word_features = self.word_vectorizer.fit_transform(materialized)
        char_features = self.char_vectorizer.fit_transform(materialized)
        return hstack([word_features, char_features], format="csr", dtype=np.float32)

    def transform(self, texts: Iterable[str]):
        materialized = self._materialize_texts(texts)
        if not materialized:
            raise ValueError("At least one text is required for TF-IDF transformation.")
        word_features = self.word_vectorizer.transform(materialized)
        char_features = self.char_vectorizer.transform(materialized)
        return hstack([word_features, char_features], format="csr", dtype=np.float32)

    def save(self, output_dir: Path | str) -> None:
        destination = ensure_dir(output_dir)
        joblib.dump(
            {
                "config": asdict(self.config),
                "word_vectorizer": self.word_vectorizer,
                "char_vectorizer": self.char_vectorizer,
            },
            Path(destination) / "tfidf_extractor.joblib",
        )
        dump_json(
            {
                "feature_type": "tfidf_word_char",
                "config": asdict(self.config),
            },
            Path(destination) / "tfidf_metadata.json",
        )

    @classmethod
    def load(cls, output_dir: Path | str) -> "TfidfFeatureExtractor":
        payload = joblib.load(Path(output_dir) / "tfidf_extractor.joblib")
        extractor = cls(TfidfFeatureConfig(**payload["config"]))
        extractor.word_vectorizer = payload["word_vectorizer"]
        extractor.char_vectorizer = payload["char_vectorizer"]
        return extractor
