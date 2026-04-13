from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd


def ensure_dir(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def reset_dir(path: Path | str, preserve_names: tuple[str, ...] = ()) -> Path:
    directory = ensure_dir(path)
    preserved = set(preserve_names)
    preserved_existing = {child.name for child in directory.iterdir() if child.name in preserved}
    for child in directory.iterdir():
        if child.name in preserved:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    for name in preserved_existing:
        placeholder = directory / name
        if not placeholder.exists():
            placeholder.touch()
    return directory


def _atomic_destination(path: Path) -> Path:
    ensure_dir(path.parent)
    temp_name = f".{path.name}.{uuid4().hex}.tmp"
    if path.suffix:
        temp_name = f".{path.stem}.{uuid4().hex}.tmp{path.suffix}"
    return path.parent / temp_name


def dump_json(data: dict, path: Path | str) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    temporary = _atomic_destination(destination)
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def load_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_table(path: Path | str) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".jsonl":
        return pd.read_json(source, orient="records", lines=True)
    if suffix == ".json":
        return pd.read_json(source)
    raise ValueError(f"Unsupported table format: {source}")


def write_table(frame: pd.DataFrame, path: Path | str) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    temporary = _atomic_destination(destination)
    suffix = destination.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination)
        return
    if suffix == ".csv":
        frame.to_csv(temporary, index=False)
        temporary.replace(destination)
        return
    if suffix == ".jsonl":
        frame.to_json(temporary, orient="records", lines=True)
        temporary.replace(destination)
        return
    if suffix == ".json":
        frame.to_json(temporary, orient="records")
        temporary.replace(destination)
        return
    raise ValueError(f"Unsupported table format: {destination}")


def timestamp_run_id(prefix: str) -> str:
    utc_now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{utc_now}"
