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

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_DEPTHS = (1, 2, 3)
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.1
DEFAULT_REQUEST_DELAY_SECONDS = 0.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
PRICING_VERIFIED_AT = "2026-04-13"
PRICING_SOURCE_URL = "https://openai.com/api/pricing/"

PARAPHRASE_ATTACK_NAME = "openai_recursive_paraphrase"
PARAPHRASE_VARIANT_PREFIX = "recursive_paraphrase_depth_"


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


def estimate_tokens(text: str) -> int:
    stripped = str(text or "").strip()
    if not stripped:
        return 0
    return max(1, int(math.ceil(len(stripped) / 4.0)))


def build_paraphrase_prompt(*, question: str, answer: str, domain: str) -> str:
    return "\n".join(
        [
            "Rewrite the AI-generated answer below as a close paraphrase.",
            "",
            "Requirements:",
            "- Preserve meaning, factual claims, sentiment, numbers, named entities, and domain terminology.",
            "- Change wording and sentence structure materially.",
            "- Return only the rewritten answer text.",
            "- Do not add explanations, bullet points, labels, or quotation marks.",
            "",
            f"Domain: {domain or 'unknown'}",
            "Question:",
            question.strip(),
            "",
            "Answer to rewrite:",
            answer.strip(),
        ]
    ).strip()


def estimate_usage_for_frame(frame: pd.DataFrame) -> UsageTotals:
    ai_rows = frame.loc[frame["label"].astype(str) == "ai"]
    usage = UsageTotals()
    usage.requests = int(len(ai_rows))
    for row in ai_rows.itertuples(index=False):
        prompt = build_paraphrase_prompt(question=str(row.prompt), answer=str(row.text), domain=str(row.domain))
        usage.input_tokens += estimate_tokens(prompt)
        usage.output_tokens += estimate_tokens(str(row.text))
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
) -> dict[str, Any]:
    base_usage = estimate_usage_for_frame(frame)
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
            depth_payload[str(depth)] = {
                "incremental_usage": incremental_usage.to_dict(),
                "cumulative_usage": cumulative_usage.to_dict(),
                "estimated_incremental_cost_usd": None if pricing is None else round(pricing.estimate_cost(incremental_usage), 6),
                "estimated_cumulative_cost_usd": None if pricing is None else round(pricing.estimate_cost(cumulative_usage), 6),
            }

        run_total_usage = UsageTotals(
            requests=base_usage.requests * requested_max_depth,
            input_tokens=base_usage.input_tokens * requested_max_depth,
            output_tokens=base_usage.output_tokens * requested_max_depth,
            total_tokens=base_usage.total_tokens * requested_max_depth,
        )
        models_payload[model] = {
            "pricing": None if pricing is None else asdict(pricing),
            "depths": depth_payload,
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


class RecursiveParaphraser:
    def __init__(
        self,
        *,
        model: str,
        request_delay_seconds: float,
        max_retries: int,
        temperature: float,
        max_output_tokens: int,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.request_delay_seconds = request_delay_seconds
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
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
        checkpoint = load_paraphrase_checkpoint(checkpoint_path)

        for row in ai_rows.itertuples(index=False):
            sample_id = str(row.sample_id)
            record = checkpoint.get(sample_id, ParaphraseRecord(sample_id=sample_id))

            completed_depth = 0
            while completed_depth + 1 in record.depth_outputs:
                completed_depth += 1
            if all(depth in record.depth_outputs for depth in requested_depths):
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
                        "depth": depth,
                        "response_id": record.response_ids_by_depth[depth],
                        "usage": incremental_usage.to_dict(),
                    },
                )

                current_text = output_text
                if self.request_delay_seconds > 0:
                    time.sleep(self.request_delay_seconds)

            append_jsonl(checkpoint_path, record.to_dict())
            checkpoint[sample_id] = record

        return checkpoint


def summarize_paraphrase_usage(
    records: dict[str, ParaphraseRecord],
    *,
    depths: list[int],
    model: str,
    pricing: PricingEntry | None,
) -> dict[str, Any]:
    max_depth = max(depths)
    incremental_usage: dict[int, UsageTotals] = {depth: UsageTotals() for depth in range(1, max_depth + 1)}
    for record in records.values():
        for depth, usage in record.incremental_usage_by_depth.items():
            incremental_usage[depth].add(usage)

    cumulative_usage: dict[int, UsageTotals] = {}
    running = UsageTotals()
    for depth in range(1, max_depth + 1):
        running.add(incremental_usage[depth])
        cumulative_usage[depth] = running.copy()

    payload: dict[str, Any] = {"model": model, "depths": {}}
    for depth in depths:
        payload["depths"][str(depth)] = {
            "incremental_usage": incremental_usage[depth].to_dict(),
            "cumulative_usage": cumulative_usage[depth].to_dict(),
            "actual_incremental_cost_usd": None if pricing is None else round(pricing.estimate_cost(incremental_usage[depth]), 6),
            "actual_cumulative_cost_usd": None if pricing is None else round(pricing.estimate_cost(cumulative_usage[depth]), 6),
        }
    payload["selected_run_total_usage"] = cumulative_usage[max_depth].to_dict()
    payload["selected_run_total_cost_usd"] = None if pricing is None else round(pricing.estimate_cost(cumulative_usage[max_depth]), 6)
    return payload


def build_paraphrased_dataset(
    control_frame: pd.DataFrame,
    paraphrase_records: dict[str, ParaphraseRecord],
    *,
    model: str,
    depth: int,
    seed: int,
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
        if depth not in record.depth_outputs:
            raise KeyError(f"Missing depth {depth} paraphrase for AI sample '{sample_id}'.")

        rewritten_text = record.depth_outputs[depth]
        source_model = str(row["source_model"])
        metadata = {
            "generator_model": model,
            "recursion_depth": depth,
            "seed": seed,
            "source_sample_id": sample_id,
            "source_model": source_model,
            "created_at": created_at,
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
) -> dict[str, Any]:
    summary = {
        "seed": seed,
        "sample_rows": sample_rows,
        "sample_fraction": sample_fraction,
        "sampled_source_rows": len(sampled_source_rows),
        "sampled_source_rows_by_domain": summarize_source_rows(sampled_source_rows)["domains"],
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
        return Path(args.output_dir).resolve()
    sample_tag = format_sample_tag(args.sample_rows, args.sample_fraction)
    return (DEFAULT_OUTPUT_ROOT / f"{args.source_file.stem}_{sample_tag}_seed{args.seed}").resolve()


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
    return parser


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    generator_models = parse_generator_models(args.generator_model)
    depths = parse_depths(args.depths)
    pricing_overrides = parse_model_pricing(args.model_pricing)

    source_rows = load_source_rows(args.source_file)
    sampled_source_rows = sample_source_rows(
        source_rows,
        sample_rows=args.sample_rows,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )
    source_summary = summarize_source_rows(source_rows)
    sample_selection = build_sample_selection_summary(
        sample_rows=args.sample_rows,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
        sampled_source_rows=sampled_source_rows,
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
            request_delay_seconds=args.request_delay_seconds,
            max_retries=args.max_retries,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
        paraphrase_records = paraphraser.paraphrase_rows(
            ai_rows,
            depths=depths,
            checkpoint_path=checkpoint_path,
            api_call_log_path=api_call_log_path,
        )
        usage_summary = summarize_paraphrase_usage(paraphrase_records, depths=depths, model=model, pricing=pricing)

        for depth in depths:
            variant_frame = build_paraphrased_dataset(
                control_frame,
                paraphrase_records,
                model=model,
                depth=depth,
                seed=args.seed,
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
                    "paraphrase_usage": usage_summary["depths"][str(depth)],
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
                }
            )

    generation_manifest = {
        "created_at": utc_now_iso(),
        "source_file": str(args.source_file),
        "source_summary": source_summary,
        "sample_selection": sample_selection,
        "estimate_report_path": str(output_dir / "estimate.json"),
        "control_dataset_dir": str(control_dataset_dir),
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
    generator_models = parse_generator_models(args.generator_model)
    depths = parse_depths(args.depths)
    pricing_overrides = parse_model_pricing(args.model_pricing)

    source_rows = load_source_rows(args.source_file)
    sampled_source_rows = sample_source_rows(
        source_rows,
        sample_rows=args.sample_rows,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )
    source_summary = summarize_source_rows(source_rows)
    sample_selection = build_sample_selection_summary(
        sample_rows=args.sample_rows,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
        sampled_source_rows=sampled_source_rows,
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
    )
    report["control_summary"] = control_summary

    output_dir = resolve_output_dir(args)
    ensure_dir(output_dir)
    write_json(output_dir / "estimate.json", report)
    return {"output_dir": str(output_dir), "estimate": report}


def main() -> None:
    load_env_file(DEFAULT_ENV_FILE)
    parser = build_parser()
    args = parser.parse_args()
    args.source_file = Path(args.source_file).resolve()
    args.output_dir = None if args.output_dir is None else Path(args.output_dir).resolve()

    if args.command == "estimate":
        result = estimate_command(args)
    elif args.command == "run":
        result = run_command(args)
    else:  # pragma: no cover - argparse keeps this unreachable.
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
