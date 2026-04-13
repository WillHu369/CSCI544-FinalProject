from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from gpt_zero.config import DEFAULT_HC3_HF_DATASET, DEFAULT_RANDOM_STATE
from gpt_zero.io_utils import dump_json, ensure_dir, read_table, write_table
from gpt_zero.schemas import LABEL_AI, LABEL_HUMAN, SAMPLE_COLUMNS, coerce_label, ensure_sample_schema
from gpt_zero.text_utils import normalize_whitespace


@dataclass
class PrepareHC3Config:
    output_dir: Path
    input_file: Path | None = None
    hf_dataset: str = DEFAULT_HC3_HF_DATASET
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = DEFAULT_RANDOM_STATE
    max_samples_per_label: int | None = None
    group_by_prompt: bool = True
    deduplicate_texts: bool = True


def _stable_sample_id(dataset: str, domain: str, prompt: str, text: str, label: str, row_index: int, answer_index: int) -> str:
    payload = "||".join([dataset, domain, prompt, text, label, str(row_index), str(answer_index)])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def _as_text_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = normalize_whitespace(value)
        return [text] if text else []
    if isinstance(value, Iterable):
        outputs: list[str] = []
        for item in value:
            candidate = item.get("text") or item.get("answer") or item.get("content") if isinstance(item, dict) else item
            text = normalize_whitespace(str(candidate or ""))
            if text:
                outputs.append(text)
        return outputs
    return []


def _extract_labelled_answers(row: dict) -> tuple[list[str], list[str], str]:
    if "text" in row and "label" in row:
        label = coerce_label(row["label"])
        text = normalize_whitespace(str(row["text"]))
        if not text:
            return [], [], "unknown"
        if label == LABEL_HUMAN:
            return [text], [], str(row.get("source_model") or "human")
        return [], [text], str(row.get("source_model") or "unknown_ai")

    human_answers = _as_text_list(
        row.get("human_answers")
        or row.get("human")
        or row.get("human_response")
        or row.get("human_text")
        or row.get("reference")
    )
    ai_answers = _as_text_list(
        row.get("chatgpt_answers")
        or row.get("ai_answers")
        or row.get("machine_answers")
        or row.get("gpt_answers")
        or row.get("model_answers")
    )
    source_model = str(row.get("source_model") or row.get("model") or "chatgpt")
    return human_answers, ai_answers, source_model


def normalize_hc3_rows(rows: list[dict], dataset_name: str = "hc3") -> pd.DataFrame:
    records: list[dict] = []
    for row_index, row in enumerate(rows):
        domain = str(row.get("domain") or row.get("source") or row.get("category") or "unknown")
        prompt = normalize_whitespace(str(row.get("question") or row.get("prompt") or row.get("instruction") or ""))
        human_answers, ai_answers, source_model = _extract_labelled_answers(row)

        for answer_index, answer in enumerate(human_answers):
            records.append(
                {
                    "sample_id": _stable_sample_id(dataset_name, domain, prompt, answer, LABEL_HUMAN, row_index, answer_index),
                    "dataset": dataset_name,
                    "domain": domain,
                    "prompt": prompt,
                    "text": answer,
                    "label": LABEL_HUMAN,
                    "source_model": "human",
                    "split": None,
                    "variant_type": "original",
                    "parent_sample_id": None,
                    "attack_name": None,
                    "attack_metadata": "{}",
                }
            )

        for answer_index, answer in enumerate(ai_answers):
            records.append(
                {
                    "sample_id": _stable_sample_id(dataset_name, domain, prompt, answer, LABEL_AI, row_index, answer_index),
                    "dataset": dataset_name,
                    "domain": domain,
                    "prompt": prompt,
                    "text": answer,
                    "label": LABEL_AI,
                    "source_model": source_model,
                    "split": None,
                    "variant_type": "original",
                    "parent_sample_id": None,
                    "attack_name": None,
                    "attack_metadata": "{}",
                }
            )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("No HC3-compatible samples were extracted from the provided data.")
    return ensure_sample_schema(frame)


def _load_hc3_via_hub_jsonl(dataset_name: str) -> list[dict]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "HC3 fallback loading requires 'huggingface_hub', which is installed with the project requirements."
        ) from exc

    jsonl_path = hf_hub_download(repo_id=dataset_name, repo_type="dataset", filename="all.jsonl")
    return read_table(jsonl_path).to_dict(orient="records")


def _load_hf_dataset(dataset_name: str) -> list[dict]:
    if dataset_name == DEFAULT_HC3_HF_DATASET:
        return _load_hc3_via_hub_jsonl(dataset_name)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The 'datasets' package is required for Hugging Face loading.") from exc

    loaded = load_dataset(dataset_name)
    rows: list[dict] = []
    if hasattr(loaded, "items"):
        for split_name, split_data in loaded.items():
            for record in split_data:
                entry = dict(record)
                entry.setdefault("domain", split_name)
                rows.append(entry)
        return rows

    for record in loaded:
        rows.append(dict(record))
    return rows


def load_hc3_rows(input_file: Path | None = None, hf_dataset: str = DEFAULT_HC3_HF_DATASET) -> list[dict]:
    if input_file is not None:
        source = Path(input_file)
        if source.suffix.lower() in {".jsonl", ".csv", ".parquet", ".json"}:
            return read_table(source).to_dict(orient="records")
        raise ValueError(f"Unsupported local dataset format: {source}")
    return _load_hf_dataset(hf_dataset)


def _apply_label_balancing(frame: pd.DataFrame, max_samples_per_label: int | None, random_state: int) -> pd.DataFrame:
    if max_samples_per_label is None:
        return frame
    return (
        frame.sample(frac=1.0, random_state=random_state)
        .groupby("label", group_keys=False)
        .head(max_samples_per_label)
        .reset_index(drop=True)
    )


def _split_with_fallback(indices: pd.Index, labels: pd.Series, test_size: float, random_state: int) -> tuple[pd.Index, pd.Index]:
    index_values = indices.to_list()
    try:
        left, right = train_test_split(index_values, test_size=test_size, random_state=random_state, stratify=labels)
    except ValueError:
        left, right = train_test_split(index_values, test_size=test_size, random_state=random_state)
    return pd.Index(left), pd.Index(right)


def _prompt_split_keys(frame: pd.DataFrame) -> pd.Series:
    prompts = frame["prompt"].fillna("").astype(str).map(normalize_whitespace)
    fallback = frame["sample_id"].astype(str)
    return prompts.where(prompts != "", fallback)


def _text_split_keys(frame: pd.DataFrame) -> pd.Series:
    return frame["text"].fillna("").astype(str).map(normalize_whitespace)


def _with_split_keys(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["_prompt_key"] = _prompt_split_keys(working)
    working["_text_key"] = _text_split_keys(working)
    return working


def duplicate_summary(frame: pd.DataFrame) -> dict[str, int]:
    working = _with_split_keys(frame)
    text_label_conflicts = working.groupby("_text_key")["label"].nunique()
    return {
        "duplicate_prompt_text_label": int(working.duplicated(subset=["_prompt_key", "_text_key", "label"]).sum()),
        "duplicate_text_label": int(working.duplicated(subset=["_text_key", "label"]).sum()),
        "duplicate_text": int(working.duplicated(subset=["_text_key"]).sum()),
        "conflicting_text_labels": int((text_label_conflicts > 1).sum()),
    }


def deduplicate_samples(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    working = _with_split_keys(frame)
    before = int(len(working))
    before_summary = duplicate_summary(working.drop(columns=["_prompt_key", "_text_key"]))
    deduplicated = working.drop_duplicates(subset=["_text_key", "label"], keep="first").reset_index(drop=True)
    after = int(len(deduplicated))
    after_summary = duplicate_summary(deduplicated.drop(columns=["_prompt_key", "_text_key"]))
    stats = {
        "rows_before": before,
        "rows_after": after,
        "rows_removed": before - after,
        "duplicates_before": before_summary["duplicate_text_label"],
        "duplicates_after": after_summary["duplicate_text_label"],
        "conflicting_text_labels": before_summary["conflicting_text_labels"],
    }
    return deduplicated.drop(columns=["_prompt_key", "_text_key"]), stats


def _assign_row_splits(frame: pd.DataFrame, test_size: float, val_size: float, random_state: int) -> pd.DataFrame:
    working = frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    train_val_indices, test_indices = _split_with_fallback(working.index, working["label"], test_size=test_size, random_state=random_state)

    remaining = working.loc[train_val_indices]
    val_ratio = 0.0 if not train_val_indices.size else val_size / (1.0 - test_size)
    if val_ratio > 0.0:
        _, val_indices = _split_with_fallback(remaining.index, remaining["label"], test_size=val_ratio, random_state=random_state)
    else:
        val_indices = pd.Index([])

    working["split"] = "train"
    working.loc[val_indices, "split"] = "val"
    working.loc[test_indices, "split"] = "test"
    return working[SAMPLE_COLUMNS]


def _mode_or_first(values: pd.Series) -> str:
    modes = values.dropna().astype(str).mode()
    if not modes.empty:
        return str(modes.iloc[0])
    if values.empty:
        return "unknown"
    return str(values.iloc[0])


def _connected_component_groups(frame: pd.DataFrame) -> pd.Series:
    working = _with_split_keys(frame)
    parent: dict[tuple[str, str], tuple[str, str]] = {}
    rank: dict[tuple[str, str], int] = {}

    def _add(node: tuple[str, str]) -> None:
        if node not in parent:
            parent[node] = node
            rank[node] = 0

    def _find(node: tuple[str, str]) -> tuple[str, str]:
        _add(node)
        if parent[node] != node:
            parent[node] = _find(parent[node])
        return parent[node]

    def _union(left: tuple[str, str], right: tuple[str, str]) -> None:
        left_root = _find(left)
        right_root = _find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    for prompt_key, text_key in zip(working["_prompt_key"], working["_text_key"], strict=False):
        _union(("prompt", str(prompt_key)), ("text", str(text_key)))

    group_ids: dict[tuple[str, str], int] = {}
    assignments: list[int] = []
    next_group_id = 0
    for prompt_key in working["_prompt_key"]:
        root = _find(("prompt", str(prompt_key)))
        if root not in group_ids:
            group_ids[root] = next_group_id
            next_group_id += 1
        assignments.append(group_ids[root])
    return pd.Series(assignments, index=frame.index, dtype="int64")


def _assign_group_splits(frame: pd.DataFrame, test_size: float, val_size: float, random_state: int) -> pd.DataFrame:
    working = _with_split_keys(frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True))
    working["_split_group"] = _connected_component_groups(working)

    group_frame = working.groupby("_split_group", as_index=False).agg(
        domain=("domain", _mode_or_first),
        label_signature=("label", lambda series: "+".join(sorted(set(series.astype(str))))),
    )
    group_frame["stratify_key"] = group_frame["domain"].astype(str) + "::" + group_frame["label_signature"].astype(str)

    train_val_groups, test_groups = _split_with_fallback(
        pd.Index(group_frame["_split_group"]),
        group_frame["stratify_key"],
        test_size=test_size,
        random_state=random_state,
    )

    remaining_groups = group_frame.loc[group_frame["_split_group"].isin(train_val_groups)]
    val_ratio = 0.0 if not train_val_groups.size else val_size / (1.0 - test_size)
    if val_ratio > 0.0:
        _, val_groups = _split_with_fallback(
            pd.Index(remaining_groups["_split_group"]),
            remaining_groups["stratify_key"],
            test_size=val_ratio,
            random_state=random_state,
        )
    else:
        val_groups = pd.Index([])

    working["split"] = "train"
    working.loc[working["_split_group"].isin(val_groups), "split"] = "val"
    working.loc[working["_split_group"].isin(test_groups), "split"] = "test"
    return working.drop(columns=["_split_group", "_prompt_key", "_text_key"])[SAMPLE_COLUMNS]


def split_overlap_summary(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    working = _with_split_keys(frame)
    prompts_by_split: dict[str, set[str]] = {}
    texts_by_split: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        split_frame = working.loc[working["split"] == split]
        prompts_by_split[split] = set(split_frame["_prompt_key"].astype(str))
        texts_by_split[split] = set(split_frame["_text_key"].astype(str))
    return {
        "prompt": {
            "train_val": int(len(prompts_by_split["train"] & prompts_by_split["val"])),
            "train_test": int(len(prompts_by_split["train"] & prompts_by_split["test"])),
            "val_test": int(len(prompts_by_split["val"] & prompts_by_split["test"])),
        },
        "text": {
            "train_val": int(len(texts_by_split["train"] & texts_by_split["val"])),
            "train_test": int(len(texts_by_split["train"] & texts_by_split["test"])),
            "val_test": int(len(texts_by_split["val"] & texts_by_split["test"])),
        },
    }


def validate_split_integrity(
    frame: pd.DataFrame,
    *,
    require_prompt_disjoint: bool,
    require_text_disjoint: bool,
    require_deduplicated: bool,
) -> dict[str, dict | int]:
    overlap = split_overlap_summary(frame)
    duplicates = duplicate_summary(frame)
    violations: list[str] = []

    if require_prompt_disjoint and any(overlap["prompt"].values()):
        violations.append(f"prompt overlap detected: {overlap['prompt']}")
    if require_text_disjoint and any(overlap["text"].values()):
        violations.append(f"text overlap detected: {overlap['text']}")
    if require_deduplicated and duplicates["duplicate_text_label"] > 0:
        violations.append(f"duplicate text+label rows remain: {duplicates['duplicate_text_label']}")
    if duplicates["conflicting_text_labels"] > 0:
        violations.append(f"texts with conflicting labels remain: {duplicates['conflicting_text_labels']}")

    if violations:
        raise ValueError("HC3 split integrity validation failed: " + "; ".join(violations))
    return {"overlap": overlap, "duplicates": duplicates}


def assign_splits(
    frame: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
    group_by_prompt: bool = True,
) -> pd.DataFrame:
    if not 0.0 <= test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0.0 <= val_size < 1.0:
        raise ValueError("val_size must be between 0 and 1.")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1.")
    if group_by_prompt:
        return _assign_group_splits(frame, test_size, val_size, random_state)
    return _assign_row_splits(frame, test_size, val_size, random_state)


def _sample_prepared_split(frame: pd.DataFrame, fraction: float, random_state: int) -> pd.DataFrame:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1.")
    if fraction >= 1.0 or frame.empty:
        return frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    working = frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    group_columns = [column for column in ("label", "domain") if column in working.columns]
    if not group_columns:
        sample_size = max(1, int(round(len(working) * fraction)))
        return working.sample(n=min(sample_size, len(working)), random_state=random_state).reset_index(drop=True)

    sampled_groups: list[pd.DataFrame] = []
    for group_index, (_, group) in enumerate(working.groupby(group_columns, dropna=False, group_keys=False)):
        sample_size = max(1, int(round(len(group) * fraction)))
        sample_size = min(sample_size, len(group))
        sampled_groups.append(group.sample(n=sample_size, random_state=random_state + group_index).reset_index(drop=True))
    return pd.concat(sampled_groups, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def sample_prepared_dataset(input_dir: Path | str, output_dir: Path | str, fraction: float, random_state: int) -> dict:
    source_dir = Path(input_dir)
    destination_dir = ensure_dir(output_dir)
    sampled_frames: list[pd.DataFrame] = []

    for split_index, split in enumerate(("train", "val", "test")):
        split_path = source_dir / f"{split}.parquet"
        if not split_path.exists():
            continue
        split_frame = read_table(split_path)
        sampled_split = _sample_prepared_split(split_frame, fraction=fraction, random_state=random_state + split_index)
        write_table(sampled_split, destination_dir / f"{split}.parquet")
        sampled_frames.append(sampled_split)

    if not sampled_frames:
        raise RuntimeError(f"No prepared split files were found under {source_dir}")

    sampled_full = pd.concat(sampled_frames, ignore_index=True)
    write_table(sampled_full, destination_dir / "full.parquet")
    integrity = validate_split_integrity(
        sampled_full,
        require_prompt_disjoint=True,
        require_text_disjoint=True,
        require_deduplicated=True,
    )

    manifest = {
        "dataset": "hc3",
        "source": str(source_dir),
        "num_samples": int(len(sampled_full)),
        "counts_by_label": sampled_full["label"].value_counts().to_dict(),
        "counts_by_split": sampled_full["split"].value_counts().to_dict(),
        "domains": sorted(sampled_full["domain"].dropna().astype(str).unique().tolist()),
        "schema": SAMPLE_COLUMNS,
        "split_strategy": "prompt_text_disjoint_fraction_sample",
        "split_overlap": integrity["overlap"],
        "duplicate_summary": integrity["duplicates"],
        "sampling": {
            "fraction": fraction,
            "random_state": random_state,
        },
    }
    dump_json(manifest, destination_dir / "manifest.json")
    return manifest


def prepare_hc3_dataset(config: PrepareHC3Config) -> dict:
    ensure_dir(config.output_dir)
    rows = load_hc3_rows(config.input_file, config.hf_dataset)
    normalized = normalize_hc3_rows(rows)
    normalized = _apply_label_balancing(normalized, config.max_samples_per_label, config.random_state)
    deduplication = {
        "rows_before": int(len(normalized)),
        "rows_after": int(len(normalized)),
        "rows_removed": 0,
        "duplicates_before": duplicate_summary(normalized)["duplicate_text_label"],
        "duplicates_after": duplicate_summary(normalized)["duplicate_text_label"],
        "conflicting_text_labels": duplicate_summary(normalized)["conflicting_text_labels"],
    }
    if config.deduplicate_texts:
        normalized, deduplication = deduplicate_samples(normalized)
    split_frame = assign_splits(
        normalized,
        config.test_size,
        config.val_size,
        config.random_state,
        group_by_prompt=config.group_by_prompt,
    )
    integrity = validate_split_integrity(
        split_frame,
        require_prompt_disjoint=config.group_by_prompt,
        require_text_disjoint=True,
        require_deduplicated=config.deduplicate_texts,
    )

    for split in ("train", "val", "test"):
        write_table(split_frame.loc[split_frame["split"] == split].reset_index(drop=True), Path(config.output_dir) / f"{split}.parquet")

    write_table(split_frame, Path(config.output_dir) / "full.parquet")

    manifest = {
        "dataset": "hc3",
        "source": str(config.input_file) if config.input_file else config.hf_dataset,
        "num_samples": int(len(split_frame)),
        "counts_by_label": split_frame["label"].value_counts().to_dict(),
        "counts_by_split": split_frame["split"].value_counts().to_dict(),
        "domains": sorted(split_frame["domain"].dropna().astype(str).unique().tolist()),
        "schema": SAMPLE_COLUMNS,
        "split_strategy": "prompt_text_disjoint" if config.group_by_prompt else "row_stratified",
        "split_overlap": integrity["overlap"],
        "duplicate_summary": integrity["duplicates"],
        "deduplication": deduplication,
        "config": {
            "test_size": config.test_size,
            "val_size": config.val_size,
            "random_state": config.random_state,
            "max_samples_per_label": config.max_samples_per_label,
            "group_by_prompt": config.group_by_prompt,
            "deduplicate_texts": config.deduplicate_texts,
        },
    }
    dump_json(manifest, Path(config.output_dir) / "manifest.json")
    return manifest
