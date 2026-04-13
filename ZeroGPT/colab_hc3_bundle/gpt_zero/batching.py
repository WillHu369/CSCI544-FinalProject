from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence, TypeAlias

import pandas as pd

from gpt_zero.io_utils import read_table


@dataclass(frozen=True)
class BatchLoaderConfig:
    batch_size: int = 1024
    columns: tuple[str, ...] | None = None


class SampleBatchLoader:
    def __init__(self, source: pd.DataFrame | Path | str, batch_size: int = 1024, columns: Sequence[str] | None = None):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.source = source
        self.config = BatchLoaderConfig(
            batch_size=int(batch_size),
            columns=tuple(columns) if columns is not None else None,
        )

    @classmethod
    def from_source(
        cls,
        source: BatchSource,
        batch_size: int = 1024,
        columns: Sequence[str] | None = None,
    ) -> "SampleBatchLoader":
        if isinstance(source, cls):
            if source.config.batch_size == batch_size and (columns is None or tuple(columns) == source.config.columns):
                return source
            return cls(source.source, batch_size=batch_size, columns=columns or source.config.columns)
        return cls(source, batch_size=batch_size, columns=columns)

    def _select_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.config.columns:
            return frame.reset_index(drop=True)
        selected = [column for column in self.config.columns if column in frame.columns]
        return frame.loc[:, selected].reset_index(drop=True)

    def _iter_frame_batches(self, frame: pd.DataFrame) -> Iterator[pd.DataFrame]:
        working = self._select_columns(frame)
        if working.empty:
            return
        for start in range(0, len(working), self.config.batch_size):
            yield working.iloc[start : start + self.config.batch_size].reset_index(drop=True)

    def _iter_parquet_batches(self, path: Path) -> Iterator[pd.DataFrame]:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            yield from self._iter_frame_batches(read_table(path))
            return

        parquet_file = pq.ParquetFile(path)
        selected_columns = list(self.config.columns) if self.config.columns is not None else None
        for batch in parquet_file.iter_batches(batch_size=self.config.batch_size, columns=selected_columns):
            yield self._select_columns(batch.to_pandas())

    def iter_frames(self) -> Iterator[pd.DataFrame]:
        if isinstance(self.source, pd.DataFrame):
            yield from self._iter_frame_batches(self.source)
            return

        path = Path(self.source)
        if path.suffix.lower() == ".parquet":
            yield from self._iter_parquet_batches(path)
            return
        yield from self._iter_frame_batches(read_table(path))

    def iter_texts(self, text_column: str = "text") -> Iterator[str]:
        for frame in self.iter_frames():
            if text_column not in frame.columns:
                continue
            for value in frame[text_column].fillna("").astype(str):
                yield value

    def num_rows(self) -> int:
        if isinstance(self.source, pd.DataFrame):
            return int(len(self.source))

        path = Path(self.source)
        if path.suffix.lower() == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError:
                return int(len(read_table(path)))
            return int(pq.ParquetFile(path).metadata.num_rows)
        return int(len(read_table(path)))

    def num_batches(self) -> int:
        total_rows = self.num_rows()
        if total_rows == 0:
            return 0
        return (total_rows + self.config.batch_size - 1) // self.config.batch_size

    def read_all(self) -> pd.DataFrame:
        frames = list(self.iter_frames())
        if not frames:
            return pd.DataFrame(columns=list(self.config.columns or []))
        return pd.concat(frames, ignore_index=True)


BatchSource: TypeAlias = pd.DataFrame | Path | str | SampleBatchLoader
