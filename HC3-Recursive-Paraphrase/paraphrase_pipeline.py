#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any

import pandas as pd

try:
    from openai import APIConnectionError, APIError, OpenAI, RateLimitError
except ImportError:  # pragma: no cover - exercised only when the SDK is missing.
    OpenAI = None

    class APIError(Exception):
        pass

    class APIConnectionError(APIError):
        pass

    class RateLimitError(APIError):
        pass


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ZEROGPT_BUNDLE_DIR = PROJECT_ROOT / "ZeroGPT" / "colab_hc3_bundle"

DEFAULT_SOURCE_FILE = PROJECT_ROOT / "HC3-Dataset-Samples" / "hc3_unified_1000_seed42.csv"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "artifacts" / "experiments"
DEFAULT_ENV_FILE = SCRIPT_DIR / ".env"
DEFAULT_PROMPT_PREFIX_FILE = SCRIPT_DIR / "prompt_prefix.txt"
DEFAULT_FINAL_DATASETS_DIR = SCRIPT_DIR / "datasets_final"

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_DEPTHS = (1, 2, 3)
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.1
DEFAULT_REQUEST_DELAY_SECONDS = 0.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_QUALITY_CHECK_DEPTH = 3
DEFAULT_QUALITY_MIN_SCORE = 3
QUALITY_CHECK_MAX_OUTPUT_TOKENS = 128
QUALITY_CHECK_ESTIMATED_OUTPUT_TOKENS = 64
PRICING_VERIFIED_AT = "2026-04-13"
PRICING_SOURCE_URL = "https://openai.com/api/pricing/"

PARAPHRASE_ATTACK_NAME = "openai_recursive_paraphrase"
PARAPHRASE_VARIANT_PREFIX = "recursive_paraphrase_depth_"
QUALITY_STATEMENT = "This document was a high quality piece of text."
LIKERT_SCORE_LABELS = {
    1: "Strongly Disagree",
    2: "Disagree",
    3: "Neutral / Neither Agree nor Disagree",
    4: "Agree",
    5: "Strongly Agree",
}
HC3_UNIFIED_COLUMNS = ("hc3_row_id", "source", "question", "human_answers", "chatgpt_answers")
HC3_FINAL_EXPORT_FILES = ("full.csv",)
HC3_STALE_SPLIT_EXPORT_FILES = ("train.csv", "val.csv", "test.csv")
DEFAULT_PROMPT_PREFIX = """
You are a high-quality paraphrasing model. Rewrite the AI-generated answer below as a faithful paraphrase that preserves the original meaning while using noticeably different wording, sentence structure, and style.

Requirements:
- Preserve all factual claims, numbers, named entities, domain terminology, uncertainty, and sentiment.
- Do not add new facts, remove important details, answer the question from scratch, or correct the source text unless the correction is purely grammatical.
- Make the rewrite substantially different from the input; do not rely on light synonym swaps or minor edits.
- Use natural human phrasing with varied rhythm and sentence structure, while keeping the answer clear and domain appropriate.
- Reduce generic assistant-style wording, repetitive transitions, polished boilerplate, and overly balanced phrasing.
- Keep the same language, approximate length, and answer format unless changing the format is necessary for fluency.
- Return only the rewritten answer text.
- Do not include explanations, labels, quotation marks, or tags.
""".strip()


@dataclass(frozen=True)
class PricingEntry:
    input_cost_per_1m_tokens: float
    output_cost_per_1m_tokens: float
    pricing_verified_at: str = PRICING_VERIFIED_AT
    pricing_source_url: str = PRICING_SOURCE_URL

    def estimate_cost(self, usage: "UsageTotals") -> float:
        return (
            usage.input_tokens * self.input_cost_per_1m_tokens
            + usage.output_tokens * self.output_cost_per_1m_tokens
        ) / 1_000_000.0


MODEL_PRICING: dict[str, PricingEntry] = {
    "gpt-5.4": PricingEntry(input_cost_per_1m_tokens=2.50, output_cost_per_1m_tokens=15.00),
    "gpt-5.4-mini": PricingEntry(input_cost_per_1m_tokens=0.75, output_cost_per_1m_tokens=4.50),
    "gpt-5.4-nano": PricingEntry(input_cost_per_1m_tokens=0.20, output_cost_per_1m_tokens=1.25),
}


@dataclass
class UsageTotals:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_response(cls, response: Any) -> "UsageTotals":
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return cls(requests=1)

        def read(name: str) -> int:
            value = getattr(usage, name, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(name)
            return int(value or 0)

        return cls(
            requests=1,
            input_tokens=read("input_tokens"),
            output_tokens=read("output_tokens"),
            total_tokens=read("total_tokens"),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "UsageTotals":
        payload = payload or {}
        return cls(
            requests=int(payload.get("requests", 0) or 0),
            input_tokens=int(payload.get("input_tokens", 0) or 0),
            output_tokens=int(payload.get("output_tokens", 0) or 0),
            total_tokens=int(payload.get("total_tokens", 0) or 0),
        )

    def add(self, other: "UsageTotals") -> None:
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens

    def copy(self) -> "UsageTotals":
        return UsageTotals(
            requests=self.requests,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ParaphraseRecord:
    sample_id: str
    depth_outputs: dict[int, str] = field(default_factory=dict)
    incremental_usage_by_depth: dict[int, UsageTotals] = field(default_factory=dict)
    cumulative_usage_by_depth: dict[int, UsageTotals] = field(default_factory=dict)
    response_ids_by_depth: dict[int, str | None] = field(default_factory=dict)
    quality_checks_by_depth: dict[int, dict[str, Any]] = field(default_factory=dict)
    quality_usage_by_depth: dict[int, UsageTotals] = field(default_factory=dict)
    quality_response_ids_by_depth: dict[int, str | None] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParaphraseRecord":
        return cls(
            sample_id=str(payload["sample_id"]),
            depth_outputs={int(key): str(value) for key, value in (payload.get("depth_outputs") or {}).items()},
            incremental_usage_by_depth={
                int(key): UsageTotals.from_dict(value)
                for key, value in (payload.get("incremental_usage_by_depth") or {}).items()
            },
            cumulative_usage_by_depth={
                int(key): UsageTotals.from_dict(value)
                for key, value in (payload.get("cumulative_usage_by_depth") or {}).items()
            },
            response_ids_by_depth={
                int(key): None if value is None else str(value)
                for key, value in (payload.get("response_ids_by_depth") or {}).items()
            },
            quality_checks_by_depth={
                int(key): dict(value) if isinstance(value, dict) else {"raw_value": value}
                for key, value in (payload.get("quality_checks_by_depth") or {}).items()
            },
            quality_usage_by_depth={
                int(key): UsageTotals.from_dict(value)
                for key, value in (payload.get("quality_usage_by_depth") or {}).items()
            },
            quality_response_ids_by_depth={
                int(key): None if value is None else str(value)
                for key, value in (payload.get("quality_response_ids_by_depth") or {}).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "depth_outputs": {str(key): value for key, value in sorted(self.depth_outputs.items())},
            "incremental_usage_by_depth": {
                str(key): value.to_dict() for key, value in sorted(self.incremental_usage_by_depth.items())
            },
            "cumulative_usage_by_depth": {
                str(key): value.to_dict() for key, value in sorted(self.cumulative_usage_by_depth.items())
            },
            "response_ids_by_depth": {
                str(key): value for key, value in sorted(self.response_ids_by_depth.items())
            },
            "quality_checks_by_depth": {
                str(key): value for key, value in sorted(self.quality_checks_by_depth.items())
            },
            "quality_usage_by_depth": {
                str(key): value.to_dict() for key, value in sorted(self.quality_usage_by_depth.items())
            },
            "quality_response_ids_by_depth": {
                str(key): value for key, value in sorted(self.quality_response_ids_by_depth.items())
            },
        }


@dataclass(frozen=True)
class HC3Helpers:
    normalize_hc3_rows: Any
    deduplicate_samples: Any
    assign_splits: Any
    validate_split_integrity: Any


_HC3_HELPERS_CACHE: HC3Helpers | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_env_file(path: Path | str, *, override: bool = False) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
                value = value[1:-1]
            if override or key not in os.environ:
                os.environ[key] = value


def load_prompt_prefix(path: Path | str = DEFAULT_PROMPT_PREFIX_FILE) -> str:
    prompt_path = Path(path)
    if not prompt_path.exists():
        return DEFAULT_PROMPT_PREFIX

    text = prompt_path.read_text(encoding="utf-8").strip()
    return text or DEFAULT_PROMPT_PREFIX


def reset_dir(path: Path | str, preserve_names: tuple[str, ...] = ()) -> Path:
    directory = ensure_dir(path)
    preserved = set(preserve_names)
    for child in list(directory.iterdir()):
        if child.name in preserved:
            continue
        if child.is_dir():
            for descendant in sorted(child.rglob("*"), reverse=True):
                if descendant.is_file():
                    descendant.unlink()
                elif descendant.is_dir():
                    descendant.rmdir()
            child.rmdir()
        else:
            child.unlink()
    for name in preserved:
        placeholder = directory / name
        if not placeholder.exists():
            placeholder.touch()
    return directory


def write_json(path: Path | str, payload: dict[str, Any]) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_json_records(path: Path | str, records: list[dict[str, Any]]) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    destination.write_text(json.dumps(records, indent=2), encoding="utf-8")


def append_jsonl(path: Path | str, payload: dict[str, Any]) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {source} on line {line_number}.") from exc
    return records


def slugify(value: str) -> str:
    characters = []
    for character in value:
        if character.isalnum():
            characters.append(character.lower())
        else:
            characters.append("_")
    slug = "".join(characters)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "value"


def stable_variant_sample_id(parent_sample_id: str, model: str, depth: int, seed: int, text: str) -> str:
    payload = "||".join([parent_sample_id, model, str(depth), str(seed), text])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def add_zerogpt_bundle_to_path() -> None:
    bundle_path = str(ZEROGPT_BUNDLE_DIR)
    if bundle_path not in sys.path:
        sys.path.insert(0, bundle_path)


def get_hc3_helpers() -> HC3Helpers:
    global _HC3_HELPERS_CACHE
    if _HC3_HELPERS_CACHE is not None:
        return _HC3_HELPERS_CACHE

    add_zerogpt_bundle_to_path()
    module = importlib.import_module("gpt_zero.hc3")
    _HC3_HELPERS_CACHE = HC3Helpers(
        normalize_hc3_rows=module.normalize_hc3_rows,
        deduplicate_samples=module.deduplicate_samples,
        assign_splits=module.assign_splits,
        validate_split_integrity=module.validate_split_integrity,
    )
    return _HC3_HELPERS_CACHE


def parse_generator_models(values: list[str] | None) -> list[str]:
    raw_values = values or [DEFAULT_MODEL]
    models: list[str] = []
    for value in raw_values:
        for item in str(value).split(","):
            candidate = item.strip()
            if candidate and candidate not in models:
                models.append(candidate)
    return models or [DEFAULT_MODEL]


def parse_depths(values: list[str] | None) -> list[int]:
    raw_values = values or [str(depth) for depth in DEFAULT_DEPTHS]
    depths: list[int] = []
    for value in raw_values:
        for item in str(value).split(","):
            candidate = item.strip()
            if not candidate:
                continue
            depth = int(candidate)
            if depth < 1:
                raise ValueError("Depth values must be positive integers.")
            if depth not in depths:
                depths.append(depth)
    return sorted(depths or list(DEFAULT_DEPTHS))


def resolve_quality_check_depth(depths: list[int], requested_depth: int | None) -> int | None:
    if requested_depth is None or requested_depth <= 0:
        return None
    if requested_depth not in depths:
        return None
    return requested_depth


def validate_quality_min_score(value: int) -> None:
    if value < 1 or value > 5:
        raise ValueError("--quality-min-score must be between 1 and 5.")


def parse_model_pricing(values: list[str] | None) -> dict[str, PricingEntry]:
    parsed: dict[str, PricingEntry] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Invalid --model-pricing value '{value}'. Expected MODEL=INPUT,OUTPUT")
        model, pricing = value.split("=", 1)
        model = model.strip()
        if not model:
            raise ValueError(f"Invalid --model-pricing value '{value}'. Model name is empty.")
        input_value, output_value = [item.strip() for item in pricing.split(",", 1)]
        parsed[model] = PricingEntry(
            input_cost_per_1m_tokens=float(input_value),
            output_cost_per_1m_tokens=float(output_value),
        )
    return parsed


def resolve_pricing(model: str, overrides: dict[str, PricingEntry]) -> PricingEntry | None:
    return overrides.get(model) or MODEL_PRICING.get(model)


def decode_list_cell(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]

    text = str(value).strip()
    if not text:
        return []

    for loader in (json.loads, ast.literal_eval):
        try:
            decoded = loader(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(decoded, list):
            return [str(item) for item in decoded if str(item).strip()]
        return [str(decoded)]
    return [text]


def encode_list_cell(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def load_source_rows(source_file: Path) -> list[dict[str, Any]]:
    if not source_file.exists():
        raise FileNotFoundError(f"Source file was not found: {source_file}")

    frame = pd.read_csv(source_file)
    required_columns = {"source", "question", "human_answers", "chatgpt_answers"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"Source file is missing required columns: {', '.join(missing)}")

    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        normalized = dict(row)
        normalized["source"] = str(normalized.get("source") or "").strip() or "unknown"
        normalized["question"] = str(normalized.get("question") or "").strip()
        normalized["human_answers"] = decode_list_cell(normalized.get("human_answers"))
        normalized["chatgpt_answers"] = decode_list_cell(normalized.get("chatgpt_answers"))
        rows.append(normalized)
    return rows


def rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if "human_answers" in frame.columns:
        frame["human_answers"] = frame["human_answers"].map(decode_list_cell).map(encode_list_cell)
    if "chatgpt_answers" in frame.columns:
        frame["chatgpt_answers"] = frame["chatgpt_answers"].map(decode_list_cell).map(encode_list_cell)
    return frame


def summarize_source_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    domain_counts = Counter(str(row.get("source") or "unknown") for row in rows)
    human_items = sum(len(decode_list_cell(row.get("human_answers"))) for row in rows)
    ai_items = sum(len(decode_list_cell(row.get("chatgpt_answers"))) for row in rows)
    return {
        "source_rows": len(rows),
        "domains": dict(sorted(domain_counts.items())),
        "human_answer_items": human_items,
        "ai_answer_items": ai_items,
    }


def allocate_domain_counts(domain_counts: dict[str, int], total: int) -> dict[str, int]:
    if total < 1:
        raise ValueError("The requested sample size must be positive.")
    available = sum(domain_counts.values())
    if total > available:
        raise ValueError(f"Requested {total} source rows, but only {available} are available.")

    exact = {domain: total * count / available for domain, count in domain_counts.items()}
    allocated = {domain: min(count, int(math.floor(exact[domain]))) for domain, count in domain_counts.items()}
    remaining = total - sum(allocated.values())
    ranked_domains = sorted(domain_counts, key=lambda domain: (-(exact[domain] - allocated[domain]), domain))

    while remaining > 0:
        changed = False
        for domain in ranked_domains:
            if allocated[domain] >= domain_counts[domain]:
                continue
            allocated[domain] += 1
            remaining -= 1
            changed = True
            if remaining == 0:
                break
        if not changed:
            break
    return allocated


def sample_source_rows(
    rows: list[dict[str, Any]],
    *,
    sample_rows: int | None,
    sample_fraction: float | None,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_rows is not None and sample_fraction is not None:
        raise ValueError("Use either --sample-rows or --sample-fraction, not both.")

    domain_to_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        domain = str(row.get("source") or "unknown")
        domain_to_rows.setdefault(domain, []).append(copy.deepcopy(row))

    if sample_rows is None and sample_fraction is None:
        return [copy.deepcopy(row) for row in rows]

    if sample_fraction is not None:
        if not 0 < sample_fraction <= 1:
            raise ValueError("--sample-fraction must be in the interval (0, 1].")
        target_total = max(1, int(round(len(rows) * sample_fraction)))
    else:
        target_total = int(sample_rows or 0)

    per_domain_targets = allocate_domain_counts(
        {domain: len(items) for domain, items in domain_to_rows.items()},
        target_total,
    )

    sampled_rows: list[dict[str, Any]] = []
    for index, domain in enumerate(sorted(domain_to_rows)):
        domain_rows = list(domain_to_rows[domain])
        Random(seed + index).shuffle(domain_rows)
        sampled_rows.extend(domain_rows[: per_domain_targets[domain]])

    def sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
        row_id = row.get("hc3_row_id")
        try:
            numeric_row_id = int(row_id)
        except (TypeError, ValueError):
            numeric_row_id = 0
        return (str(row.get("source") or "unknown"), numeric_row_id, str(row.get("question") or ""))

    return sorted(sampled_rows, key=sort_key)


def validate_shard_args(num_shards: int, shard_index: int) -> None:
    if num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if shard_index < 1 or shard_index > num_shards:
        raise ValueError(f"--shard-index must be in [1, {num_shards}], got {shard_index}.")


def shard_tag(num_shards: int, shard_index: int) -> str:
    width = max(2, len(str(num_shards)))
    return f"shard_{shard_index:0{width}d}_of_{num_shards:0{width}d}"


def shard_source_rows(rows: list[dict[str, Any]], *, num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    validate_shard_args(num_shards, shard_index)
    if num_shards == 1:
        return [copy.deepcopy(row) for row in rows]

    zero_based_shard = shard_index - 1
    domain_to_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        domain = str(row.get("source") or "unknown")
        domain_to_rows.setdefault(domain, []).append(row)

    selected: list[dict[str, Any]] = []
    for domain in sorted(domain_to_rows):
        for row_index, row in enumerate(domain_to_rows[domain]):
            if row_index % num_shards == zero_based_shard:
                selected.append(copy.deepcopy(row))

    return selected


def prepare_control_frame(
    source_rows: list[dict[str, Any]],
    *,
    random_state: int,
    test_size: float,
    val_size: float,
    hc3_helpers: HC3Helpers | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    helpers = hc3_helpers or get_hc3_helpers()
    normalized = helpers.normalize_hc3_rows(source_rows, dataset_name="hc3")
    deduplicated, deduplication = helpers.deduplicate_samples(normalized)
    split_frame = helpers.assign_splits(
        deduplicated,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        group_by_prompt=True,
    )
    integrity = helpers.validate_split_integrity(
        split_frame,
        require_prompt_disjoint=True,
        require_text_disjoint=True,
        require_deduplicated=True,
    )
    ordered = split_frame.sort_values(["split", "domain", "label", "sample_id"]).reset_index(drop=True)
    manifest = {
        "num_samples": int(len(ordered)),
        "counts_by_label": ordered["label"].value_counts().to_dict(),
        "counts_by_split": ordered["split"].value_counts().to_dict(),
        "counts_by_domain": ordered["domain"].value_counts().to_dict(),
        "deduplication": deduplication,
        "split_overlap": integrity["overlap"],
        "duplicate_summary": integrity["duplicates"],
    }
    return ordered, manifest


def build_dataset_manifest(
    frame: pd.DataFrame,
    *,
    dataset_kind: str,
    source_file: Path,
    source_summary: dict[str, Any],
    sample_selection: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "created_at": utc_now_iso(),
        "dataset_kind": dataset_kind,
        "source_file": str(source_file),
        "source_summary": source_summary,
        "sample_selection": sample_selection,
        "num_samples": int(len(frame)),
        "counts_by_label": frame["label"].value_counts().to_dict(),
        "counts_by_split": frame["split"].value_counts().to_dict(),
        "counts_by_domain": frame["domain"].value_counts().to_dict(),
    }
    if extra:
        payload.update(extra)
    return payload


def write_dataset_bundle(dataset_dir: Path, frame: pd.DataFrame, manifest: dict[str, Any]) -> None:
    dataset_dir = ensure_dir(dataset_dir)
    for split in ("train", "val", "test"):
        split_frame = frame.loc[frame["split"] == split].reset_index(drop=True)
        split_frame.to_csv(dataset_dir / f"{split}.csv", index=False)
    frame.reset_index(drop=True).to_csv(dataset_dir / "full.csv", index=False)
    write_json(dataset_dir / "manifest.json", manifest)


def clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def normalize_group_value(value: Any) -> str:
    return " ".join(clean_scalar(value).split())


def hc3_row_key(source: Any, question: Any) -> tuple[str, str]:
    return (normalize_group_value(source), normalize_group_value(question))


def load_export_source_rows(experiment_dir: Path, source_file: Path) -> tuple[list[dict[str, Any]], Path]:
    sampled_source_file = experiment_dir / "sampled_source_rows.csv"
    selected_source_file = sampled_source_file if sampled_source_file.exists() else source_file
    return load_source_rows(selected_source_file), selected_source_file


def nonempty_answer_list(value: Any) -> list[str]:
    return [answer for answer in decode_list_cell(value) if normalize_group_value(answer)]


def build_hc3_source_lookup(source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    helpers = get_hc3_helpers()
    normalized_samples = helpers.normalize_hc3_rows(source_rows, dataset_name="hc3").reset_index(drop=True)
    source_payloads: list[dict[str, Any]] = []
    sample_id_to_row_index: dict[str, int] = {}
    sample_id_to_ai_answer_order: dict[str, int] = {}
    sample_id_to_ai_answer_text: dict[str, str] = {}
    key_to_row_indices: dict[tuple[str, str], list[int]] = {}
    cursor = 0

    for row_index, row in enumerate(source_rows):
        key = hc3_row_key(row.get("source"), row.get("question"))
        human_answers = nonempty_answer_list(row.get("human_answers"))
        ai_answers = nonempty_answer_list(row.get("chatgpt_answers"))
        source_payloads.append(
            {
                "order": row_index,
                "key": key,
                "hc3_row_id": clean_scalar(row.get("hc3_row_id")),
                "source": clean_scalar(row.get("source")),
                "question": clean_scalar(row.get("question")),
                "human_answers": human_answers,
                "chatgpt_answers": ai_answers,
            }
        )
        key_to_row_indices.setdefault(key, []).append(row_index)

        for _ in human_answers:
            if cursor >= len(normalized_samples):
                raise ValueError("HC3 source normalization produced fewer rows than expected while mapping human answers.")
            sample = normalized_samples.iloc[cursor]
            if clean_scalar(sample.get("label")) != "human":
                raise ValueError(
                    "Unexpected HC3 normalized row order while mapping human answers. "
                    f"Expected human, got {sample.get('label')!r} at normalized row {cursor}."
                )
            sample_id_to_row_index[clean_scalar(sample.get("sample_id"))] = row_index
            cursor += 1

        for answer_order, _ in enumerate(ai_answers):
            if cursor >= len(normalized_samples):
                raise ValueError("HC3 source normalization produced fewer rows than expected while mapping AI answers.")
            sample = normalized_samples.iloc[cursor]
            if clean_scalar(sample.get("label")) != "ai":
                raise ValueError(
                    "Unexpected HC3 normalized row order while mapping AI answers. "
                    f"Expected ai, got {sample.get('label')!r} at normalized row {cursor}."
                )
            sample_id = clean_scalar(sample.get("sample_id"))
            sample_id_to_row_index[sample_id] = row_index
            sample_id_to_ai_answer_order[sample_id] = answer_order
            sample_id_to_ai_answer_text[sample_id] = ai_answers[answer_order]
            cursor += 1

    if cursor != len(normalized_samples):
        raise ValueError(
            f"HC3 source normalization mapping consumed {cursor} rows, "
            f"but normalized source contains {len(normalized_samples)} rows."
        )

    return {
        "rows": source_payloads,
        "sample_id_to_row_index": sample_id_to_row_index,
        "sample_id_to_ai_answer_order": sample_id_to_ai_answer_order,
        "sample_id_to_ai_answer_text": sample_id_to_ai_answer_text,
        "key_to_row_indices": key_to_row_indices,
    }


def hc3_source_row_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        clean_scalar(row.get("hc3_row_id")),
        normalize_group_value(row.get("source")),
        normalize_group_value(row.get("question")),
    )


def build_hc3_source_lookup_for_experiment(experiment_dir: Path, source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    base_lookup = build_hc3_source_lookup(source_rows)
    generation_manifest_path = experiment_dir / "generation_manifest.json"
    if not generation_manifest_path.exists():
        return base_lookup

    try:
        generation_manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return base_lookup

    shard_manifests = generation_manifest.get("merged_shard_manifests") or []
    if not shard_manifests:
        return base_lookup

    row_identity_to_index = {
        hc3_source_row_identity(source_payload): row_index
        for row_index, source_payload in enumerate(base_lookup["rows"])
    }
    sample_id_to_row_index: dict[str, int] = {}
    sample_id_to_ai_answer_order: dict[str, int] = {}
    sample_id_to_ai_answer_text: dict[str, str] = {}

    for shard_manifest in shard_manifests:
        shard_dir_text = shard_manifest.get("_shard_dir")
        if not shard_dir_text:
            continue
        shard_source_path = Path(shard_dir_text) / "sampled_source_rows.csv"
        if not shard_source_path.exists():
            continue
        shard_source_rows = load_source_rows(shard_source_path)
        shard_lookup = build_hc3_source_lookup(shard_source_rows)

        shard_row_to_global_index: dict[int, int] = {}
        for shard_row_index, shard_payload in enumerate(shard_lookup["rows"]):
            identity = hc3_source_row_identity(shard_payload)
            if identity in row_identity_to_index:
                shard_row_to_global_index[shard_row_index] = row_identity_to_index[identity]

        for sample_id, shard_row_index in shard_lookup["sample_id_to_row_index"].items():
            if shard_row_index in shard_row_to_global_index:
                sample_id_to_row_index[sample_id] = shard_row_to_global_index[shard_row_index]
        for sample_id, answer_order in shard_lookup["sample_id_to_ai_answer_order"].items():
            sample_id_to_ai_answer_order[sample_id] = answer_order
        for sample_id, answer_text in shard_lookup["sample_id_to_ai_answer_text"].items():
            sample_id_to_ai_answer_text[sample_id] = answer_text

    if not sample_id_to_row_index:
        return base_lookup

    merged_lookup = dict(base_lookup)
    merged_lookup["sample_id_to_row_index"] = sample_id_to_row_index
    merged_lookup["sample_id_to_ai_answer_order"] = sample_id_to_ai_answer_order
    merged_lookup["sample_id_to_ai_answer_text"] = sample_id_to_ai_answer_text
    return merged_lookup


def build_legacy_hc3_source_lookup(source_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row_index, row in enumerate(source_rows):
        key = hc3_row_key(row.get("source"), row.get("question"))
        lookup.setdefault(
            key,
            {
            "order": row_index,
            "hc3_row_id": clean_scalar(row.get("hc3_row_id")),
            "source": clean_scalar(row.get("source")),
            "question": clean_scalar(row.get("question")),
            "human_answers": decode_list_cell(row.get("human_answers")),
            },
        )
    return lookup


def build_ai_sample_order(source_rows: list[dict[str, Any]]) -> dict[str, int]:
    helpers = get_hc3_helpers()
    normalized = helpers.normalize_hc3_rows(source_rows, dataset_name="hc3")
    deduplicated, _ = helpers.deduplicate_samples(normalized)
    ai_rows = deduplicated.loc[deduplicated["label"].astype(str) == "ai"].reset_index(drop=True)
    return {str(row["sample_id"]): int(index) for index, row in ai_rows.iterrows()}


def discover_generated_dataset_dirs(experiment_dir: Path) -> list[tuple[str, str, Path]]:
    datasets_root = experiment_dir / "datasets"
    if not datasets_root.exists():
        raise FileNotFoundError(f"Generated datasets directory was not found: {datasets_root}")

    discovered: list[tuple[str, str, Path]] = []
    for model_dir in sorted(child for child in datasets_root.iterdir() if child.is_dir()):
        if model_dir.name == "control":
            continue
        for depth_dir in sorted(child for child in model_dir.iterdir() if child.is_dir()):
            if (depth_dir / "full.csv").exists():
                discovered.append((model_dir.name, depth_dir.name, depth_dir))

    if not discovered:
        raise FileNotFoundError(f"No generated model depth datasets were found under: {datasets_root}")
    return discovered


def convert_detector_frame_to_hc3(
    frame: pd.DataFrame,
    *,
    source_lookup: dict[str, Any],
) -> pd.DataFrame:
    required_columns = {"domain", "prompt", "text", "label", "sample_id"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"Generated dataset is missing required columns: {', '.join(missing)}")

    source_rows = source_lookup["rows"]
    sample_id_to_row_index = source_lookup["sample_id_to_row_index"]
    sample_id_to_ai_answer_order = source_lookup["sample_id_to_ai_answer_order"]
    sample_id_to_ai_answer_text = source_lookup.get("sample_id_to_ai_answer_text", {})
    key_to_row_indices = source_lookup["key_to_row_indices"]
    grouped_ai_answers: dict[int, list[tuple[int, str]]] = {}
    fallback_ai_answers_by_original_text: dict[str, list[tuple[int, str]]] = {}
    missing_source_rows: list[tuple[str, str, str]] = []

    for row_position, row in enumerate(frame.to_dict(orient="records")):
        key = hc3_row_key(row.get("domain"), row.get("prompt"))
        label = clean_scalar(row.get("label")).lower()
        parent_sample_id = clean_scalar(row.get("parent_sample_id"))
        sample_id = clean_scalar(row.get("sample_id"))
        source_sample_id = parent_sample_id if label == "ai" and parent_sample_id else sample_id
        row_index = sample_id_to_row_index.get(source_sample_id)

        if row_index is None:
            candidate_indices = key_to_row_indices.get(key, [])
            if len(candidate_indices) == 1:
                row_index = candidate_indices[0]
            else:
                missing_source_rows.append((source_sample_id, key[0], key[1]))
                continue

        if label != "ai":
            continue

        answer_order = sample_id_to_ai_answer_order.get(source_sample_id, row_position)
        text = clean_scalar(row.get("text")).strip()
        if text:
            grouped_ai_answers.setdefault(row_index, []).append((answer_order, text))
            original_answer_key = normalize_group_value(sample_id_to_ai_answer_text.get(source_sample_id))
            if original_answer_key:
                fallback_ai_answers_by_original_text.setdefault(original_answer_key, []).append((answer_order, text))

    if missing_source_rows:
        preview = ", ".join(
            f"{sample_id or '<empty>'} / {source}: {question[:60]}"
            for sample_id, source, question in missing_source_rows[:5]
        )
        raise ValueError(f"Generated rows could not be matched to source HC3 rows: {preview}")

    def ordered_unique_texts(items: list[tuple[int, str]]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for _, text in sorted(items, key=lambda item: item[0]):
            normalized = normalize_group_value(text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            output.append(text)
        return output

    records: list[dict[str, Any]] = []
    for row_index, source_payload in enumerate(source_rows):
        ai_answers = ordered_unique_texts(grouped_ai_answers.get(row_index, []))
        if not ai_answers and source_payload["chatgpt_answers"]:
            reused_answers: list[str] = []
            reused_keys: set[str] = set()
            for answer_order, original_answer in enumerate(source_payload["chatgpt_answers"]):
                original_answer_key = normalize_group_value(original_answer)
                if not original_answer_key or original_answer_key in reused_keys:
                    continue
                reused_keys.add(original_answer_key)
                candidates = ordered_unique_texts(fallback_ai_answers_by_original_text.get(original_answer_key, []))
                if candidates:
                    reused_answers.append(candidates[0])
            ai_answers = reused_answers
        if not ai_answers:
            ai_answers = source_payload["chatgpt_answers"]
        records.append(
            {
                "hc3_row_id": source_payload["hc3_row_id"],
                "source": source_payload["source"],
                "question": source_payload["question"],
                "human_answers": encode_list_cell(source_payload["human_answers"]),
                "chatgpt_answers": encode_list_cell(ai_answers),
            }
        )

    return pd.DataFrame.from_records(records, columns=list(HC3_UNIFIED_COLUMNS))


def export_dataset_dir_to_hc3(
    source_dataset_dir: Path,
    destination_dir: Path,
    *,
    source_lookup: dict[str, Any],
) -> dict[str, Any]:
    ensure_dir(destination_dir)
    for stale_filename in HC3_STALE_SPLIT_EXPORT_FILES:
        stale_file = destination_dir / stale_filename
        if stale_file.exists():
            stale_file.unlink()

    exported_files: dict[str, dict[str, Any]] = {}
    for filename in HC3_FINAL_EXPORT_FILES:
        source_file = source_dataset_dir / filename
        if not source_file.exists():
            continue
        frame = pd.read_csv(source_file)
        converted = convert_detector_frame_to_hc3(
            frame,
            source_lookup=source_lookup,
        )
        destination_file = destination_dir / filename
        converted.to_csv(destination_file, index=False)
        exported_files[filename] = {
            "path": str(destination_file),
            "rows": int(len(converted)),
            "columns": list(converted.columns),
        }

    manifest = {
        "created_at": utc_now_iso(),
        "source_dataset_dir": str(source_dataset_dir),
        "destination_dir": str(destination_dir),
        "files": exported_files,
        "format": "hc3_unified",
        "columns": list(HC3_UNIFIED_COLUMNS),
    }
    write_json(destination_dir / "manifest.json", manifest)
    return manifest


def estimate_tokens(text: str) -> int:
    stripped = str(text or "").strip()
    if not stripped:
        return 0
    return max(1, int(math.ceil(len(stripped) / 4.0)))


def build_paraphrase_prompt(*, question: str, answer: str, domain: str, prompt_prefix: str) -> str:
    return "\n".join(
        [
            prompt_prefix.strip(),
            "",
            f"Domain: {domain or 'unknown'}",
            "Question:",
            question.strip(),
            "",
            "Answer to rewrite:",
            answer.strip(),
        ]
    ).strip()


def build_quality_check_prompt(text: str) -> str:
    scale_lines = [LIKERT_SCORE_LABELS[score] for score in range(1, 6)]
    return "\n".join(
        [
            "Evaluate the document against the statement and Likert scale below.",
            "",
            f'Statement: "{QUALITY_STATEMENT}"',
            *scale_lines,
            "",
            "Use this numeric mapping in your answer:",
            "1 = Strongly Disagree",
            "2 = Disagree",
            "3 = Neutral / Neither Agree nor Disagree",
            "4 = Agree",
            "5 = Strongly Agree",
            "",
            'Return JSON only with these keys: "score", "label", and "reason".',
            "The score must be an integer from 1 to 5, and the label must be one of the five labels above.",
            "",
            "Document:",
            text.strip(),
        ]
    ).strip()


def normalize_likert_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def score_from_likert_label(value: Any) -> int | None:
    normalized = normalize_likert_label(value)
    label_lookup = {normalize_likert_label(label): score for score, label in LIKERT_SCORE_LABELS.items()}
    label_lookup["neutral"] = 3
    label_lookup["neither agree nor disagree"] = 3
    return label_lookup.get(normalized)


def parse_quality_check_response(output_text: str) -> dict[str, Any]:
    text = output_text.strip()
    payload: dict[str, Any] = {}
    if text:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    payload = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    payload = {}

    score: int | None = None
    raw_score = payload.get("score") or payload.get("rating") or payload.get("likert_score")
    if raw_score is not None:
        match = re.search(r"\b([1-5])\b", str(raw_score))
        if match:
            score = int(match.group(1))

    raw_label = payload.get("label") or payload.get("answer") or payload.get("choice")
    if score is None and raw_label is not None:
        score = score_from_likert_label(raw_label)

    if score is None:
        match = re.search(r"\b([1-5])\b", text)
        if match:
            score = int(match.group(1))

    if score is None:
        for label_score, label in LIKERT_SCORE_LABELS.items():
            if normalize_likert_label(label) in normalize_likert_label(text):
                score = label_score
                break

    if score is None or score < 1 or score > 5:
        raise ValueError(f"Could not parse a 1-5 Likert quality score from: {output_text!r}")

    label = str(raw_label or LIKERT_SCORE_LABELS[score])
    if score_from_likert_label(label) is None:
        label = LIKERT_SCORE_LABELS[score]

    return {
        "score": score,
        "label": label,
        "reason": str(payload.get("reason") or "").strip(),
        "raw_output": output_text,
    }


def build_quality_check_result(output_text: str, *, min_score: int) -> dict[str, Any]:
    parsed = parse_quality_check_response(output_text)
    score = int(parsed["score"])
    return {
        "statement": QUALITY_STATEMENT,
        "scale": {str(score): label for score, label in LIKERT_SCORE_LABELS.items()},
        "score": score,
        "label": parsed["label"],
        "reason": parsed["reason"],
        "min_score": min_score,
        "passed": score >= min_score,
        "checked_at": utc_now_iso(),
        "raw_output": parsed["raw_output"],
    }


def estimate_usage_for_frame(frame: pd.DataFrame, *, prompt_prefix: str) -> UsageTotals:
    ai_rows = frame.loc[frame["label"].astype(str) == "ai"]
    usage = UsageTotals()
    usage.requests = int(len(ai_rows))
    for row in ai_rows.itertuples(index=False):
        prompt = build_paraphrase_prompt(
            question=str(row.prompt),
            answer=str(row.text),
            domain=str(row.domain),
            prompt_prefix=prompt_prefix,
        )
        usage.input_tokens += estimate_tokens(prompt)
        usage.output_tokens += estimate_tokens(str(row.text))
    usage.total_tokens = usage.input_tokens + usage.output_tokens
    return usage


def estimate_quality_check_usage_for_frame(frame: pd.DataFrame) -> UsageTotals:
    ai_rows = frame.loc[frame["label"].astype(str) == "ai"]
    usage = UsageTotals()
    usage.requests = int(len(ai_rows))
    for row in ai_rows.itertuples(index=False):
        prompt = build_quality_check_prompt(str(row.text))
        usage.input_tokens += estimate_tokens(prompt)
        usage.output_tokens += QUALITY_CHECK_ESTIMATED_OUTPUT_TOKENS
    usage.total_tokens = usage.input_tokens + usage.output_tokens
    return usage


def build_estimate_report(
    frame: pd.DataFrame,
    *,
    generator_models: list[str],
    depths: list[int],
    pricing_overrides: dict[str, PricingEntry],
    sample_selection: dict[str, Any],
    source_summary: dict[str, Any],
    prompt_prefix: str,
    quality_check_depth: int | None,
    quality_min_score: int,
) -> dict[str, Any]:
    base_usage = estimate_usage_for_frame(frame, prompt_prefix=prompt_prefix)
    quality_usage = UsageTotals()
    if quality_check_depth is not None:
        quality_usage = estimate_quality_check_usage_for_frame(frame)
    requested_max_depth = max(depths)
    models_payload: dict[str, Any] = {}

    for model in generator_models:
        pricing = resolve_pricing(model, pricing_overrides)
        depth_payload: dict[str, Any] = {}
        for depth in depths:
            cumulative_usage = UsageTotals(
                requests=base_usage.requests * depth,
                input_tokens=base_usage.input_tokens * depth,
                output_tokens=base_usage.output_tokens * depth,
                total_tokens=base_usage.total_tokens * depth,
            )
            incremental_usage = UsageTotals(
                requests=base_usage.requests,
                input_tokens=base_usage.input_tokens,
                output_tokens=base_usage.output_tokens,
                total_tokens=base_usage.total_tokens,
            )
            incremental_paraphrase_usage = incremental_usage.copy()
            cumulative_paraphrase_usage = cumulative_usage.copy()
            depth_quality_usage = UsageTotals()
            if quality_check_depth is not None and depth == quality_check_depth:
                depth_quality_usage = quality_usage.copy()
                incremental_usage.add(depth_quality_usage)
            if quality_check_depth is not None and depth >= quality_check_depth:
                cumulative_usage.add(quality_usage)
            depth_payload[str(depth)] = {
                "incremental_usage": incremental_usage.to_dict(),
                "cumulative_usage": cumulative_usage.to_dict(),
                "paraphrase_incremental_usage": incremental_paraphrase_usage.to_dict(),
                "paraphrase_cumulative_usage": cumulative_paraphrase_usage.to_dict(),
                "quality_check_usage": depth_quality_usage.to_dict(),
                "estimated_incremental_cost_usd": None if pricing is None else round(pricing.estimate_cost(incremental_usage), 6),
                "estimated_cumulative_cost_usd": None if pricing is None else round(pricing.estimate_cost(cumulative_usage), 6),
                "estimated_quality_check_cost_usd": None if pricing is None else round(pricing.estimate_cost(depth_quality_usage), 6),
            }

        run_total_usage = UsageTotals(
            requests=base_usage.requests * requested_max_depth,
            input_tokens=base_usage.input_tokens * requested_max_depth,
            output_tokens=base_usage.output_tokens * requested_max_depth,
            total_tokens=base_usage.total_tokens * requested_max_depth,
        )
        if quality_check_depth is not None and requested_max_depth >= quality_check_depth:
            run_total_usage.add(quality_usage)
        models_payload[model] = {
            "pricing": None if pricing is None else asdict(pricing),
            "depths": depth_payload,
            "quality_gate": {
                "enabled": quality_check_depth is not None,
                "checked_depth": quality_check_depth,
                "min_score": quality_min_score,
                "statement": QUALITY_STATEMENT,
                "estimated_usage": quality_usage.to_dict(),
            },
            "selected_run_total_usage": run_total_usage.to_dict(),
            "selected_run_total_cost_usd": None if pricing is None else round(pricing.estimate_cost(run_total_usage), 6),
        }

    total_estimated_cost = 0.0
    any_missing_pricing = False
    for model in generator_models:
        pricing = resolve_pricing(model, pricing_overrides)
        if pricing is None:
            any_missing_pricing = True
            continue
        total_estimated_cost += float(models_payload[model]["selected_run_total_cost_usd"] or 0.0)

    return {
        "created_at": utc_now_iso(),
        "pricing_verified_at": PRICING_VERIFIED_AT,
        "pricing_source_url": PRICING_SOURCE_URL,
        "source_summary": source_summary,
        "sample_selection": sample_selection,
        "sampled_dataset_summary": {
            "num_samples": int(len(frame)),
            "counts_by_label": frame["label"].value_counts().to_dict(),
            "counts_by_split": frame["split"].value_counts().to_dict(),
            "counts_by_domain": frame["domain"].value_counts().to_dict(),
        },
        "per_depth_base_usage": base_usage.to_dict(),
        "quality_gate": {
            "enabled": quality_check_depth is not None,
            "checked_depth": quality_check_depth,
            "min_score": quality_min_score,
            "statement": QUALITY_STATEMENT,
            "estimated_usage": quality_usage.to_dict(),
        },
        "models": models_payload,
        "selected_run_total_estimated_cost_usd": None if any_missing_pricing else round(total_estimated_cost, 6),
        "missing_pricing_models": sorted(
            model for model in generator_models if resolve_pricing(model, pricing_overrides) is None
        ),
    }


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")

    chunks: list[str] = []
    for item in output or []:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        for part in content or []:
            part_type = getattr(part, "type", None)
            if part_type is None and isinstance(part, dict):
                part_type = part.get("type")
            if part_type != "output_text":
                continue
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text:
                chunks.append(str(text))
    return "\n".join(chunks).strip()


def load_paraphrase_checkpoint(checkpoint_path: Path) -> dict[str, ParaphraseRecord]:
    records: dict[str, ParaphraseRecord] = {}
    for payload in read_jsonl(checkpoint_path):
        record = ParaphraseRecord.from_dict(payload)
        records[record.sample_id] = record
    return records


def completed_consecutive_depth(record: ParaphraseRecord) -> int:
    completed_depth = 0
    while completed_depth + 1 in record.depth_outputs:
        completed_depth += 1
    return completed_depth


def print_call_progress(*, completed: int, total: int, model: str, force: bool = False) -> None:
    if total <= 0:
        if force:
            print(f"[paraphrase:{model}] checkpoint already complete")
        return
    if not force and completed % 10 != 0:
        return
    print(f"[paraphrase:{model}] calls {completed}/{total}")


class RecursiveParaphraser:
    def __init__(
        self,
        *,
        model: str,
        prompt_prefix: str,
        request_delay_seconds: float,
        max_retries: int,
        temperature: float,
        max_output_tokens: int,
        quality_check_depth: int | None,
        quality_min_score: int,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.prompt_prefix = prompt_prefix
        self.request_delay_seconds = request_delay_seconds
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.quality_check_depth = quality_check_depth
        self.quality_min_score = quality_min_score
        self.client = client or self._build_client()

    def _build_client(self) -> Any:
        if OpenAI is None:
            raise RuntimeError(
                "The OpenAI Python SDK is not installed. Install the dependencies in HC3-Recursive-Paraphrase/requirements.txt."
            )
        load_env_file(DEFAULT_ENV_FILE)
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        return OpenAI()

    def _dynamic_max_output_tokens(self, text: str) -> int:
        return max(self.max_output_tokens, estimate_tokens(text) + 128)

    def _create_response(self, prompt: str, source_text: str) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    text={"format": {"type": "text"}},
                    max_output_tokens=self._dynamic_max_output_tokens(source_text),
                    temperature=self.temperature,
                    store=False,
                )
            except (APIConnectionError, APIError, RateLimitError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(30.0, 2.0 ** (attempt - 1)))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unexpected paraphrase request failure.")

    def _create_quality_response(self, prompt: str) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    text={"format": {"type": "text"}},
                    max_output_tokens=QUALITY_CHECK_MAX_OUTPUT_TOKENS,
                    temperature=0.0,
                    store=False,
                )
            except (APIConnectionError, APIError, RateLimitError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(30.0, 2.0 ** (attempt - 1)))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unexpected quality check request failure.")

    def paraphrase_rows(
        self,
        ai_rows: pd.DataFrame,
        *,
        depths: list[int],
        checkpoint_path: Path,
        api_call_log_path: Path,
    ) -> dict[str, ParaphraseRecord]:
        requested_depths = sorted(depths)
        max_depth = max(requested_depths)
        quality_depth = self.quality_check_depth if self.quality_check_depth in requested_depths else None
        checkpoint = load_paraphrase_checkpoint(checkpoint_path)
        rows = list(ai_rows.itertuples(index=False))

        total_pending_calls = 0
        for row in rows:
            sample_id = str(row.sample_id)
            record = checkpoint.get(sample_id, ParaphraseRecord(sample_id=sample_id))
            depths_complete = all(depth in record.depth_outputs for depth in requested_depths)
            quality_complete = quality_depth is None or quality_depth in record.quality_checks_by_depth
            if depths_complete and quality_complete:
                continue
            total_pending_calls += max(0, max_depth - completed_consecutive_depth(record))
            if quality_depth is not None and quality_depth not in record.quality_checks_by_depth:
                total_pending_calls += 1

        completed_calls = 0
        print_call_progress(completed=completed_calls, total=total_pending_calls, model=self.model, force=True)

        for row in rows:
            sample_id = str(row.sample_id)
            record = checkpoint.get(sample_id, ParaphraseRecord(sample_id=sample_id))

            completed_depth = completed_consecutive_depth(record)
            depths_complete = all(depth in record.depth_outputs for depth in requested_depths)
            quality_complete = quality_depth is None or quality_depth in record.quality_checks_by_depth
            if depths_complete and quality_complete:
                continue

            current_text = str(row.text)
            cumulative_usage = UsageTotals()
            if completed_depth:
                current_text = record.depth_outputs[completed_depth]
                cumulative_usage = record.cumulative_usage_by_depth[completed_depth].copy()

            for depth in range(completed_depth + 1, max_depth + 1):
                prompt = build_paraphrase_prompt(
                    question=str(row.prompt),
                    answer=current_text,
                    domain=str(row.domain),
                    prompt_prefix=self.prompt_prefix,
                )
                response = self._create_response(prompt, current_text)
                output_text = extract_response_text(response)
                if not output_text:
                    raise RuntimeError(
                        f"Empty paraphrase output for sample {sample_id} at depth {depth} using model {self.model}."
                    )

                incremental_usage = UsageTotals.from_response(response)
                cumulative_usage.add(incremental_usage)

                record.depth_outputs[depth] = output_text
                record.incremental_usage_by_depth[depth] = incremental_usage.copy()
                record.cumulative_usage_by_depth[depth] = cumulative_usage.copy()
                record.response_ids_by_depth[depth] = getattr(response, "id", None)

                append_jsonl(
                    api_call_log_path,
                    {
                        "timestamp": utc_now_iso(),
                        "sample_id": sample_id,
                        "model": self.model,
                        "call_type": "paraphrase",
                        "depth": depth,
                        "response_id": record.response_ids_by_depth[depth],
                        "usage": incremental_usage.to_dict(),
                    },
                )

                current_text = output_text
                completed_calls += 1
                print_call_progress(completed=completed_calls, total=total_pending_calls, model=self.model)
                if self.request_delay_seconds > 0:
                    time.sleep(self.request_delay_seconds)

            if quality_depth is not None and quality_depth in record.depth_outputs and quality_depth not in record.quality_checks_by_depth:
                quality_prompt = build_quality_check_prompt(record.depth_outputs[quality_depth])
                response = self._create_quality_response(quality_prompt)
                output_text = extract_response_text(response)
                if not output_text:
                    raise RuntimeError(
                        f"Empty quality check output for sample {sample_id} at depth {quality_depth} using model {self.model}."
                    )

                quality_result = build_quality_check_result(output_text, min_score=self.quality_min_score)
                quality_usage = UsageTotals.from_response(response)
                record.quality_checks_by_depth[quality_depth] = quality_result
                record.quality_usage_by_depth[quality_depth] = quality_usage
                record.quality_response_ids_by_depth[quality_depth] = getattr(response, "id", None)

                append_jsonl(
                    api_call_log_path,
                    {
                        "timestamp": utc_now_iso(),
                        "sample_id": sample_id,
                        "model": self.model,
                        "call_type": "quality_check",
                        "depth": quality_depth,
                        "response_id": record.quality_response_ids_by_depth[quality_depth],
                        "statement": QUALITY_STATEMENT,
                        "min_score": self.quality_min_score,
                        "score": quality_result["score"],
                        "passed": quality_result["passed"],
                        "usage": quality_usage.to_dict(),
                    },
                )

                completed_calls += 1
                print_call_progress(completed=completed_calls, total=total_pending_calls, model=self.model)
                if self.request_delay_seconds > 0:
                    time.sleep(self.request_delay_seconds)

            append_jsonl(checkpoint_path, record.to_dict())
            checkpoint[sample_id] = record

        if total_pending_calls and (completed_calls == 0 or completed_calls % 10 != 0):
            print_call_progress(completed=completed_calls, total=total_pending_calls, model=self.model, force=True)
        return checkpoint


def summarize_quality_gate(
    records: dict[str, ParaphraseRecord],
    *,
    quality_check_depth: int | None,
    quality_min_score: int,
) -> dict[str, Any]:
    if quality_check_depth is None:
        return {
            "enabled": False,
            "checked_depth": None,
            "min_score": quality_min_score,
            "statement": QUALITY_STATEMENT,
        }

    scores: Counter[int] = Counter()
    checked = 0
    passed = 0
    missing = 0
    for record in records.values():
        result = record.quality_checks_by_depth.get(quality_check_depth)
        if not result:
            missing += 1
            continue
        checked += 1
        score = int(result.get("score") or 0)
        if score:
            scores[score] += 1
        passed_value = result.get("passed")
        passed_result = passed_value if isinstance(passed_value, bool) else score >= quality_min_score
        if passed_result:
            passed += 1

    rejected = checked - passed
    return {
        "enabled": True,
        "checked_depth": quality_check_depth,
        "min_score": quality_min_score,
        "statement": QUALITY_STATEMENT,
        "checked": checked,
        "passed": passed,
        "rejected": rejected,
        "missing": missing,
        "score_counts": {str(score): scores.get(score, 0) for score in range(1, 6)},
    }


def summarize_paraphrase_usage(
    records: dict[str, ParaphraseRecord],
    *,
    depths: list[int],
    model: str,
    pricing: PricingEntry | None,
    quality_check_depth: int | None,
    quality_min_score: int,
) -> dict[str, Any]:
    max_depth = max(depths)
    incremental_usage: dict[int, UsageTotals] = {depth: UsageTotals() for depth in range(1, max_depth + 1)}
    quality_usage_by_depth: dict[int, UsageTotals] = {depth: UsageTotals() for depth in range(1, max_depth + 1)}
    for record in records.values():
        for depth, usage in record.incremental_usage_by_depth.items():
            incremental_usage[depth].add(usage)
        for depth, usage in record.quality_usage_by_depth.items():
            if depth in quality_usage_by_depth:
                quality_usage_by_depth[depth].add(usage)

    cumulative_usage: dict[int, UsageTotals] = {}
    cumulative_paraphrase_usage: dict[int, UsageTotals] = {}
    running = UsageTotals()
    running_paraphrase = UsageTotals()
    for depth in range(1, max_depth + 1):
        running_paraphrase.add(incremental_usage[depth])
        cumulative_paraphrase_usage[depth] = running_paraphrase.copy()
        running.add(incremental_usage[depth])
        running.add(quality_usage_by_depth[depth])
        cumulative_usage[depth] = running.copy()

    quality_summary = summarize_quality_gate(records, quality_check_depth=quality_check_depth, quality_min_score=quality_min_score)
    payload: dict[str, Any] = {"model": model, "depths": {}, "quality_gate": quality_summary}
    for depth in depths:
        total_incremental_usage = incremental_usage[depth].copy()
        total_incremental_usage.add(quality_usage_by_depth[depth])
        payload["depths"][str(depth)] = {
            "incremental_usage": total_incremental_usage.to_dict(),
            "cumulative_usage": cumulative_usage[depth].to_dict(),
            "paraphrase_incremental_usage": incremental_usage[depth].to_dict(),
            "paraphrase_cumulative_usage": cumulative_paraphrase_usage[depth].to_dict(),
            "quality_check_usage": quality_usage_by_depth[depth].to_dict(),
            "actual_incremental_cost_usd": None if pricing is None else round(pricing.estimate_cost(total_incremental_usage), 6),
            "actual_cumulative_cost_usd": None if pricing is None else round(pricing.estimate_cost(cumulative_usage[depth]), 6),
            "actual_quality_check_cost_usd": None if pricing is None else round(pricing.estimate_cost(quality_usage_by_depth[depth]), 6),
        }
    payload["selected_run_total_usage"] = cumulative_usage[max_depth].to_dict()
    payload["selected_run_total_cost_usd"] = None if pricing is None else round(pricing.estimate_cost(cumulative_usage[max_depth]), 6)
    return payload


def select_paraphrased_text_for_export(
    *,
    record: ParaphraseRecord,
    requested_depth: int,
    original_text: str,
    quality_check_depth: int | None,
    quality_min_score: int,
) -> tuple[str, dict[str, Any]]:
    if requested_depth not in record.depth_outputs:
        raise KeyError(f"Missing depth {requested_depth} paraphrase for AI sample '{record.sample_id}'.")

    selected_text = record.depth_outputs[requested_depth]
    selected_depth = requested_depth
    gate_metadata: dict[str, Any] = {
        "enabled": False,
        "checked_depth": quality_check_depth,
        "min_score": quality_min_score,
        "selected_depth": selected_depth,
    }

    if quality_check_depth is None or requested_depth != quality_check_depth:
        return selected_text, gate_metadata

    quality_result = record.quality_checks_by_depth.get(quality_check_depth)
    if quality_result is None:
        raise KeyError(
            f"Missing quality check for depth {quality_check_depth} paraphrase on AI sample '{record.sample_id}'."
        )

    score = int(quality_result.get("score") or 0)
    passed_value = quality_result.get("passed")
    passed = passed_value if isinstance(passed_value, bool) else score >= quality_min_score
    rejected = not passed
    fallback_depth: int | None = None
    if rejected:
        for candidate_depth in range(requested_depth - 1, 0, -1):
            if candidate_depth in record.depth_outputs:
                fallback_depth = candidate_depth
                selected_depth = candidate_depth
                selected_text = record.depth_outputs[candidate_depth]
                break
        if fallback_depth is None:
            fallback_depth = 0
            selected_depth = 0
            selected_text = original_text

    gate_metadata.update(
        {
            "enabled": True,
            "statement": QUALITY_STATEMENT,
            "score": score,
            "label": quality_result.get("label"),
            "passed": passed,
            "rejected": rejected,
            "fallback_depth": fallback_depth,
            "selected_depth": selected_depth,
        }
    )
    return selected_text, gate_metadata


def build_paraphrased_dataset(
    control_frame: pd.DataFrame,
    paraphrase_records: dict[str, ParaphraseRecord],
    *,
    model: str,
    depth: int,
    seed: int,
    quality_check_depth: int | None,
    quality_min_score: int,
    created_at: str | None = None,
) -> pd.DataFrame:
    created_at = created_at or utc_now_iso()
    variant = control_frame.copy()
    ai_mask = variant["label"].astype(str) == "ai"

    for index, row in variant.loc[ai_mask].iterrows():
        sample_id = str(row["sample_id"])
        if sample_id not in paraphrase_records:
            raise KeyError(f"Missing paraphrase results for AI sample '{sample_id}'.")
        record = paraphrase_records[sample_id]

        rewritten_text, quality_gate_metadata = select_paraphrased_text_for_export(
            record=record,
            requested_depth=depth,
            original_text=str(row["text"]),
            quality_check_depth=quality_check_depth,
            quality_min_score=quality_min_score,
        )
        source_model = str(row["source_model"])
        metadata = {
            "generator_model": model,
            "recursion_depth": depth,
            "seed": seed,
            "source_sample_id": sample_id,
            "source_model": source_model,
            "created_at": created_at,
            "quality_gate": quality_gate_metadata,
        }

        variant.at[index, "text"] = rewritten_text
        variant.at[index, "sample_id"] = stable_variant_sample_id(sample_id, model, depth, seed, rewritten_text)
        variant.at[index, "source_model"] = model
        variant.at[index, "variant_type"] = f"{PARAPHRASE_VARIANT_PREFIX}{depth}"
        variant.at[index, "parent_sample_id"] = sample_id
        variant.at[index, "attack_name"] = PARAPHRASE_ATTACK_NAME
        variant.at[index, "attack_metadata"] = json.dumps(metadata, sort_keys=True)

    return variant.sort_values(["split", "domain", "label", "sample_id"]).reset_index(drop=True)


def build_sample_selection_summary(
    *,
    sample_rows: int | None,
    sample_fraction: float | None,
    seed: int,
    sampled_source_rows: list[dict[str, Any]],
    pre_shard_source_rows: int | None = None,
    num_shards: int = 1,
    shard_index: int = 1,
) -> dict[str, Any]:
    summary = {
        "seed": seed,
        "sample_rows": sample_rows,
        "sample_fraction": sample_fraction,
        "sampled_source_rows": len(sampled_source_rows),
        "sampled_source_rows_by_domain": summarize_source_rows(sampled_source_rows)["domains"],
        "pre_shard_source_rows": pre_shard_source_rows if pre_shard_source_rows is not None else len(sampled_source_rows),
        "num_shards": num_shards,
        "shard_index": shard_index,
        "shard_tag": shard_tag(num_shards, shard_index),
    }
    if sample_rows is None and sample_fraction is None:
        summary["selection_mode"] = "full_source"
    elif sample_rows is not None:
        summary["selection_mode"] = "balanced_source_rows"
    else:
        summary["selection_mode"] = "balanced_source_fraction"
    return summary


def format_sample_tag(sample_rows: int | None, sample_fraction: float | None) -> str:
    if sample_rows is not None:
        return f"rows{sample_rows}"
    if sample_fraction is not None:
        return f"frac{str(sample_fraction).replace('.', 'p')}"
    return "full"


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        base = Path(args.output_dir).resolve()
    else:
        sample_tag = format_sample_tag(args.sample_rows, args.sample_fraction)
        base = (DEFAULT_OUTPUT_ROOT / f"{args.source_file.stem}_{sample_tag}_seed{args.seed}").resolve()

    num_shards = int(getattr(args, "num_shards", 1) or 1)
    shard_index = int(getattr(args, "shard_index", 1) or 1)
    if num_shards > 1:
        return base / "shards" / shard_tag(num_shards, shard_index)
    return base


def validate_budget_guard(estimate_report: dict[str, Any], max_estimated_cost_usd: float | None) -> None:
    if max_estimated_cost_usd is None:
        return

    total_cost = estimate_report.get("selected_run_total_estimated_cost_usd")
    missing_pricing = estimate_report.get("missing_pricing_models") or []
    if missing_pricing:
        raise RuntimeError(
            "Cannot enforce --max-estimated-cost-usd because pricing is missing for: "
            + ", ".join(sorted(missing_pricing))
            + ". Supply --model-pricing MODEL=INPUT,OUTPUT."
        )
    if total_cost is not None and float(total_cost) > max_estimated_cost_usd:
        raise RuntimeError(
            f"Estimated OpenAI API cost ${total_cost:.4f} exceeds the configured budget guard "
            f"${max_estimated_cost_usd:.4f}."
        )


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-file", type=Path, default=DEFAULT_SOURCE_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--generator-model", action="append", default=None, help="Repeat or pass comma-separated model IDs.")
    parser.add_argument("--depths", action="append", default=None, help="Repeat or pass comma-separated recursion depths.")
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--sample-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument(
        "--quality-check-depth",
        type=int,
        default=DEFAULT_QUALITY_CHECK_DEPTH,
        help="Depth to score with the OpenAI Likert quality gate. Use 0 to disable; skipped if that depth is not generated.",
    )
    parser.add_argument(
        "--quality-min-score",
        type=int,
        default=DEFAULT_QUALITY_MIN_SCORE,
        help="Minimum 1-5 Likert score required to use the checked depth in exported datasets.",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Split sampled source rows into this many shards.")
    parser.add_argument(
        "--shard-index",
        type=int,
        default=1,
        help="One-based shard index for this process. Used with --num-shards.",
    )
    parser.add_argument(
        "--model-pricing",
        action="append",
        default=None,
        metavar="MODEL=INPUT,OUTPUT",
        help="Optional pricing override per model in USD per 1M tokens.",
    )


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-estimated-cost-usd", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--request-delay-seconds", type=float, default=DEFAULT_REQUEST_DELAY_SECONDS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)


def add_export_hc3_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Experiment directory containing the generated datasets folder.",
    )
    parser.add_argument(
        "--source-file",
        type=Path,
        default=DEFAULT_SOURCE_FILE,
        help="Fallback HC3 unified source CSV. sampled_source_rows.csv in the experiment is used when present.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FINAL_DATASETS_DIR,
        help="Destination directory for HC3 unified exports.",
    )


def add_merge_shards_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--shards-dir",
        type=Path,
        required=True,
        help="Directory containing shard_*_of_* experiment directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination experiment directory for the merged shard outputs.",
    )
    parser.add_argument("--source-file", type=Path, default=DEFAULT_SOURCE_FILE)
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build recursive paraphrase HC3 dataset variants for the existing ZeroGPT Colab workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate_parser = subparsers.add_parser("estimate", help="Estimate token usage and OpenAI API cost.")
    add_shared_arguments(estimate_parser)

    run_parser = subparsers.add_parser(
        "run",
        help="Generate recursive paraphrases and write dataset bundles only.",
    )
    add_shared_arguments(run_parser)
    add_run_arguments(run_parser)

    export_parser = subparsers.add_parser(
        "export-hc3",
        help="Convert generated answer-level datasets back to HC3 unified CSV format.",
    )
    add_export_hc3_arguments(export_parser)

    merge_parser = subparsers.add_parser(
        "merge-shards",
        help="Merge completed shard experiment directories into one experiment directory.",
    )
    add_merge_shards_arguments(merge_parser)
    return parser


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    validate_shard_args(args.num_shards, args.shard_index)
    generator_models = parse_generator_models(args.generator_model)
    depths = parse_depths(args.depths)
    validate_quality_min_score(args.quality_min_score)
    quality_check_depth = resolve_quality_check_depth(depths, args.quality_check_depth)
    pricing_overrides = parse_model_pricing(args.model_pricing)
    prompt_prefix = load_prompt_prefix()

    source_rows = load_source_rows(args.source_file)
    pre_shard_source_rows = sample_source_rows(
        source_rows,
        sample_rows=args.sample_rows,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )
    sampled_source_rows = shard_source_rows(
        pre_shard_source_rows,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    source_summary = summarize_source_rows(source_rows)
    sample_selection = build_sample_selection_summary(
        sample_rows=args.sample_rows,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
        sampled_source_rows=sampled_source_rows,
        pre_shard_source_rows=len(pre_shard_source_rows),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    control_frame, control_summary = prepare_control_frame(
        sampled_source_rows,
        random_state=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    estimate_report = build_estimate_report(
        control_frame,
        generator_models=generator_models,
        depths=depths,
        pricing_overrides=pricing_overrides,
        sample_selection=sample_selection,
        source_summary=source_summary,
        prompt_prefix=prompt_prefix,
        quality_check_depth=quality_check_depth,
        quality_min_score=args.quality_min_score,
    )
    validate_budget_guard(estimate_report, args.max_estimated_cost_usd)

    output_dir = resolve_output_dir(args)
    if args.overwrite:
        reset_dir(output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "api_calls")
    ensure_dir(output_dir / "checkpoints")

    write_json(output_dir / "estimate.json", estimate_report)
    rows_to_frame(sampled_source_rows).to_csv(output_dir / "sampled_source_rows.csv", index=False)

    control_dataset_dir = output_dir / "datasets" / "control"
    control_manifest = build_dataset_manifest(
        control_frame,
        dataset_kind="control",
        source_file=args.source_file,
        source_summary=source_summary,
        sample_selection=sample_selection,
        extra={"control_summary": control_summary},
    )
    write_dataset_bundle(control_dataset_dir, control_frame, control_manifest)

    variant_manifests: list[dict[str, Any]] = []
    ai_rows = control_frame.loc[control_frame["label"].astype(str) == "ai"].reset_index(drop=True)

    for model in generator_models:
        model_slug = slugify(model)
        pricing = resolve_pricing(model, pricing_overrides)
        checkpoint_path = output_dir / "checkpoints" / f"{model_slug}_depthmax{max(depths)}.jsonl"
        api_call_log_path = output_dir / "api_calls" / f"{model_slug}_depthmax{max(depths)}.jsonl"

        paraphraser = RecursiveParaphraser(
            model=model,
            prompt_prefix=prompt_prefix,
            request_delay_seconds=args.request_delay_seconds,
            max_retries=args.max_retries,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            quality_check_depth=quality_check_depth,
            quality_min_score=args.quality_min_score,
        )
        paraphrase_records = paraphraser.paraphrase_rows(
            ai_rows,
            depths=depths,
            checkpoint_path=checkpoint_path,
            api_call_log_path=api_call_log_path,
        )
        usage_summary = summarize_paraphrase_usage(
            paraphrase_records,
            depths=depths,
            model=model,
            pricing=pricing,
            quality_check_depth=quality_check_depth,
            quality_min_score=args.quality_min_score,
        )

        for depth in depths:
            variant_frame = build_paraphrased_dataset(
                control_frame,
                paraphrase_records,
                model=model,
                depth=depth,
                seed=args.seed,
                quality_check_depth=quality_check_depth,
                quality_min_score=args.quality_min_score,
            )
            variant_dataset_dir = output_dir / "datasets" / model_slug / f"depth_{depth}"
            variant_manifest = build_dataset_manifest(
                variant_frame,
                dataset_kind="paraphrased",
                source_file=args.source_file,
                source_summary=source_summary,
                sample_selection=sample_selection,
                extra={
                    "generator_model": model,
                    "recursion_depth": depth,
                    "attack_name": PARAPHRASE_ATTACK_NAME,
                    "control_dataset_dir": str(control_dataset_dir),
                    "prompt_prefix_file": DEFAULT_PROMPT_PREFIX_FILE.name,
                    "paraphrase_usage": usage_summary["depths"][str(depth)],
                    "quality_gate": usage_summary["quality_gate"],
                    "checkpoint_path": str(checkpoint_path),
                    "api_call_log_path": str(api_call_log_path),
                },
            )
            write_dataset_bundle(variant_dataset_dir, variant_frame, variant_manifest)

            variant_manifests.append(
                {
                    "generator_model": model,
                    "recursion_depth": depth,
                    "dataset_dir": str(variant_dataset_dir),
                    "paraphrase_usage": usage_summary["depths"][str(depth)],
                    "quality_gate": usage_summary["quality_gate"],
                }
            )

    generation_manifest = {
        "created_at": utc_now_iso(),
        "source_file": str(args.source_file),
        "source_summary": source_summary,
        "sample_selection": sample_selection,
        "estimate_report_path": str(output_dir / "estimate.json"),
        "control_dataset_dir": str(control_dataset_dir),
        "prompt_prefix_file": DEFAULT_PROMPT_PREFIX_FILE.name,
        "quality_gate": {
            "enabled": quality_check_depth is not None,
            "checked_depth": quality_check_depth,
            "min_score": args.quality_min_score,
            "statement": QUALITY_STATEMENT,
        },
        "variants": variant_manifests,
        "pricing_verified_at": PRICING_VERIFIED_AT,
        "pricing_source_url": PRICING_SOURCE_URL,
    }
    write_json(output_dir / "generation_manifest.json", generation_manifest)

    return {
        "output_dir": str(output_dir),
        "estimate": estimate_report,
        "generation_manifest": generation_manifest,
    }


def estimate_command(args: argparse.Namespace) -> dict[str, Any]:
    validate_shard_args(args.num_shards, args.shard_index)
    generator_models = parse_generator_models(args.generator_model)
    depths = parse_depths(args.depths)
    validate_quality_min_score(args.quality_min_score)
    quality_check_depth = resolve_quality_check_depth(depths, args.quality_check_depth)
    pricing_overrides = parse_model_pricing(args.model_pricing)
    prompt_prefix = load_prompt_prefix()

    source_rows = load_source_rows(args.source_file)
    pre_shard_source_rows = sample_source_rows(
        source_rows,
        sample_rows=args.sample_rows,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )
    sampled_source_rows = shard_source_rows(
        pre_shard_source_rows,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    source_summary = summarize_source_rows(source_rows)
    sample_selection = build_sample_selection_summary(
        sample_rows=args.sample_rows,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
        sampled_source_rows=sampled_source_rows,
        pre_shard_source_rows=len(pre_shard_source_rows),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    control_frame, control_summary = prepare_control_frame(
        sampled_source_rows,
        random_state=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    report = build_estimate_report(
        control_frame,
        generator_models=generator_models,
        depths=depths,
        pricing_overrides=pricing_overrides,
        sample_selection=sample_selection,
        source_summary=source_summary,
        prompt_prefix=prompt_prefix,
        quality_check_depth=quality_check_depth,
        quality_min_score=args.quality_min_score,
    )
    report["control_summary"] = control_summary

    output_dir = resolve_output_dir(args)
    ensure_dir(output_dir)
    write_json(output_dir / "estimate.json", report)
    return {"output_dir": str(output_dir), "estimate": report}


def export_hc3_command(args: argparse.Namespace) -> dict[str, Any]:
    experiment_dir = Path(args.experiment_dir)
    output_dir = ensure_dir(args.output_dir)

    source_rows, source_rows_file = load_export_source_rows(experiment_dir, args.source_file)
    source_lookup = build_hc3_source_lookup_for_experiment(experiment_dir, source_rows)

    exported_datasets: list[dict[str, Any]] = []
    for model_slug, depth_name, dataset_dir in discover_generated_dataset_dirs(experiment_dir):
        destination_dir = output_dir / model_slug / depth_name
        manifest = export_dataset_dir_to_hc3(
            dataset_dir,
            destination_dir,
            source_lookup=source_lookup,
        )
        exported_datasets.append(
            {
                "model": model_slug,
                "depth": depth_name,
                "source_dataset_dir": str(dataset_dir),
                "destination_dir": str(destination_dir),
                "files": manifest["files"],
            }
        )

    export_manifest = {
        "created_at": utc_now_iso(),
        "experiment_dir": str(experiment_dir),
        "source_rows_file": str(source_rows_file),
        "output_dir": str(output_dir),
        "format": "hc3_unified",
        "columns": list(HC3_UNIFIED_COLUMNS),
        "datasets": exported_datasets,
    }
    write_json(output_dir / "manifest.json", export_manifest)
    return {"output_dir": str(output_dir), "export_manifest": export_manifest}


def discover_shard_experiment_dirs(shards_dir: Path) -> list[Path]:
    if not shards_dir.exists():
        raise FileNotFoundError(f"Shard directory does not exist: {shards_dir}")
    shard_dirs = sorted(
        child
        for child in shards_dir.iterdir()
        if child.is_dir() and child.name.startswith("shard_") and (child / "generation_manifest.json").exists()
    )
    if not shard_dirs:
        raise FileNotFoundError(f"No completed shard experiment directories found under: {shards_dir}")
    return shard_dirs


def read_shard_manifests(shard_dirs: list[Path]) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        manifest_path = shard_dir / "generation_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["_shard_dir"] = str(shard_dir)
        manifests.append(payload)
    return manifests


def sort_hc3_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "hc3_row_id" in output.columns:
        output["_numeric_hc3_row_id"] = pd.to_numeric(output["hc3_row_id"], errors="coerce").fillna(-1)
        sort_columns = ["source", "_numeric_hc3_row_id", "question"]
    else:
        sort_columns = [column for column in ("source", "question") if column in output.columns]
    if sort_columns:
        output = output.sort_values(sort_columns).reset_index(drop=True)
    if "_numeric_hc3_row_id" in output.columns:
        output = output.drop(columns=["_numeric_hc3_row_id"])
    return output


def merge_sampled_source_rows(shard_dirs: list[Path], destination: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for shard_dir in shard_dirs:
        source_path = shard_dir / "sampled_source_rows.csv"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing sampled source rows for shard: {source_path}")
        frames.append(pd.read_csv(source_path))

    merged = pd.concat(frames, ignore_index=True)
    dedupe_columns = [column for column in ("hc3_row_id", "source", "question") if column in merged.columns]
    if dedupe_columns:
        merged = merged.drop_duplicates(subset=dedupe_columns, keep="first")
    merged = sort_hc3_source_frame(merged)
    merged.to_csv(destination, index=False)
    return merged


def merge_dataset_bundle_from_shards(
    shard_dirs: list[Path],
    *,
    relative_dataset_dir: Path,
    destination_dir: Path,
    manifest: dict[str, Any],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for shard_dir in shard_dirs:
        source_path = shard_dir / relative_dataset_dir / "full.csv"
        if source_path.exists():
            frames.append(pd.read_csv(source_path))
        else:
            missing.append(str(source_path))
    if missing:
        raise FileNotFoundError("Missing shard dataset files:\n" + "\n".join(missing))
    if not frames:
        raise RuntimeError(f"No shard dataset frames found for {relative_dataset_dir}")

    merged = pd.concat(frames, ignore_index=True)
    if "sample_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["sample_id"], keep="first")
    if {"split", "domain", "label", "sample_id"}.issubset(merged.columns):
        merged = merged.sort_values(["split", "domain", "label", "sample_id"]).reset_index(drop=True)
    else:
        merged = merged.reset_index(drop=True)
    manifest = dict(manifest)
    manifest["num_samples"] = int(len(merged))
    if "label" in merged.columns:
        manifest["counts_by_label"] = merged["label"].value_counts().to_dict()
    if "split" in merged.columns:
        manifest["counts_by_split"] = merged["split"].value_counts().to_dict()
    if "domain" in merged.columns:
        manifest["counts_by_domain"] = merged["domain"].value_counts().to_dict()
    write_dataset_bundle(destination_dir, merged, manifest)
    return merged


def discover_variant_relative_dirs(shard_dirs: list[Path]) -> list[Path]:
    relative_dirs: set[Path] = set()
    for shard_dir in shard_dirs:
        datasets_dir = shard_dir / "datasets"
        if not datasets_dir.exists():
            continue
        for model_dir in datasets_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name == "control":
                continue
            for depth_dir in model_dir.iterdir():
                if depth_dir.is_dir() and (depth_dir / "full.csv").exists():
                    relative_dirs.add(Path("datasets") / model_dir.name / depth_dir.name)
    if not relative_dirs:
        raise FileNotFoundError("No generated variant datasets were found in the shard directories.")
    return sorted(relative_dirs, key=lambda path: str(path))


def merge_shard_logs(shard_dirs: list[Path], output_dir: Path) -> dict[str, list[str]]:
    merged_paths: dict[str, list[str]] = {"api_calls": [], "checkpoints": []}
    for subdir_name in ("api_calls", "checkpoints"):
        destination_subdir = ensure_dir(output_dir / subdir_name)
        grouped: dict[str, list[Path]] = {}
        for shard_dir in shard_dirs:
            shard_subdir = shard_dir / subdir_name
            if not shard_subdir.exists():
                continue
            for jsonl_path in shard_subdir.glob("*.jsonl"):
                grouped.setdefault(jsonl_path.name, []).append(jsonl_path)

        for filename, paths in sorted(grouped.items()):
            destination = destination_subdir / filename
            with destination.open("w", encoding="utf-8") as handle:
                for path in sorted(paths):
                    text = path.read_text(encoding="utf-8")
                    if text:
                        handle.write(text)
                        if not text.endswith("\n"):
                            handle.write("\n")
            merged_paths[subdir_name].append(str(destination))
    return merged_paths


def merge_shards_command(args: argparse.Namespace) -> dict[str, Any]:
    shard_dirs = discover_shard_experiment_dirs(Path(args.shards_dir))
    output_dir = Path(args.output_dir)
    if args.overwrite:
        reset_dir(output_dir)
    ensure_dir(output_dir)

    shard_manifests = read_shard_manifests(shard_dirs)
    sampled_source_frame = merge_sampled_source_rows(shard_dirs, output_dir / "sampled_source_rows.csv")
    sampled_source_rows = load_source_rows(output_dir / "sampled_source_rows.csv")
    source_rows = load_source_rows(args.source_file)
    source_summary = summarize_source_rows(source_rows)
    sample_selection = {
        "selection_mode": "merged_shards",
        "merged_shards": len(shard_dirs),
        "sampled_source_rows": int(len(sampled_source_frame)),
        "sampled_source_rows_by_domain": summarize_source_rows(sampled_source_rows)["domains"],
        "shard_dirs": [str(path) for path in shard_dirs],
    }

    merged_logs = merge_shard_logs(shard_dirs, output_dir)

    control_frame = merge_dataset_bundle_from_shards(
        shard_dirs,
        relative_dataset_dir=Path("datasets") / "control",
        destination_dir=output_dir / "datasets" / "control",
        manifest={
            "created_at": utc_now_iso(),
            "dataset_kind": "control",
            "source_file": str(args.source_file),
            "source_summary": source_summary,
            "sample_selection": sample_selection,
            "num_samples": None,
            "merged_from_shards": [str(path) for path in shard_dirs],
        },
    )

    variant_manifests: list[dict[str, Any]] = []
    for relative_dir in discover_variant_relative_dirs(shard_dirs):
        parts = relative_dir.parts
        model_slug = parts[1]
        depth_name = parts[2]
        destination_dir = output_dir / relative_dir
        variant_frame = merge_dataset_bundle_from_shards(
            shard_dirs,
            relative_dataset_dir=relative_dir,
            destination_dir=destination_dir,
            manifest={
                "created_at": utc_now_iso(),
                "dataset_kind": "paraphrased",
                "source_file": str(args.source_file),
                "source_summary": source_summary,
                "sample_selection": sample_selection,
                "num_samples": None,
                "model_slug": model_slug,
                "recursion_depth": depth_name.replace("depth_", ""),
                "attack_name": PARAPHRASE_ATTACK_NAME,
                "merged_from_shards": [str(path) for path in shard_dirs],
            },
        )
        variant_manifests.append(
            {
                "model_slug": model_slug,
                "recursion_depth": depth_name.replace("depth_", ""),
                "dataset_dir": str(destination_dir),
                "num_samples": int(len(variant_frame)),
            }
        )

    generation_manifest = {
        "created_at": utc_now_iso(),
        "source_file": str(args.source_file),
        "source_summary": source_summary,
        "sample_selection": sample_selection,
        "control_dataset_dir": str(output_dir / "datasets" / "control"),
        "control_num_samples": int(len(control_frame)),
        "prompt_prefix_file": DEFAULT_PROMPT_PREFIX_FILE.name,
        "variants": variant_manifests,
        "merged_shard_manifests": shard_manifests,
        "merged_logs": merged_logs,
        "pricing_verified_at": PRICING_VERIFIED_AT,
        "pricing_source_url": PRICING_SOURCE_URL,
    }
    write_json(output_dir / "generation_manifest.json", generation_manifest)
    return {"output_dir": str(output_dir), "generation_manifest": generation_manifest}


def main() -> None:
    load_env_file(DEFAULT_ENV_FILE)
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "source_file"):
        args.source_file = Path(args.source_file).resolve()
    if hasattr(args, "output_dir"):
        args.output_dir = None if args.output_dir is None else Path(args.output_dir).resolve()
    if hasattr(args, "experiment_dir"):
        args.experiment_dir = Path(args.experiment_dir).resolve()
    if hasattr(args, "shards_dir"):
        args.shards_dir = Path(args.shards_dir).resolve()

    if args.command == "estimate":
        result = estimate_command(args)
    elif args.command == "run":
        result = run_command(args)
    elif args.command == "export-hc3":
        result = export_hc3_command(args)
    elif args.command == "merge-shards":
        result = merge_shards_command(args)
    else:  # pragma: no cover - argparse keeps this unreachable.
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
