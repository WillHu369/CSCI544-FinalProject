#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from huggingface_hub import hf_hub_download


OUTPUT_COLUMNS = ["hc3_row_id", "source", "question", "human_answers", "chatgpt_answers"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a balanced sample from the Hello-SimpleAI/HC3 dataset "
            "and export both a unified CSV and per-domain CSVs."
        )
    )
    parser.add_argument(
        "--samples-per-domain",
        type=int,
        default=200,
        help="Number of rows to sample from each domain.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for domain sampling.",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        help=(
            "Create one unified sample with this many total rows instead of a "
            "fixed number per domain."
        ),
    )
    parser.add_argument(
        "--preserve-unified-sample",
        type=Path,
        help=(
            "CSV whose hc3_row_id values must be included in the unified sample. "
            "Use with --total-samples when expanding an existing sample."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where the CSV files will be written.",
    )
    return parser.parse_args()


def load_hc3() -> pd.DataFrame:
    dataset_path = hf_hub_download(
        repo_id="Hello-SimpleAI/HC3",
        filename="all.jsonl",
        repo_type="dataset",
    )
    df = pd.read_json(dataset_path, lines=True)
    df.insert(0, "hc3_row_id", range(len(df)))
    return df[OUTPUT_COLUMNS]


def serialize_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ["human_answers", "chatgpt_answers"]:
        out[column] = out[column].apply(
            lambda value: json.dumps(value, ensure_ascii=False)
        )
    return out


def read_preserved_sample(path: Path) -> pd.DataFrame:
    preserved = pd.read_csv(path)
    missing_columns = sorted(set(OUTPUT_COLUMNS).difference(preserved.columns))
    if missing_columns:
        raise ValueError(
            f"Preserved sample is missing required columns: {missing_columns}"
        )
    if preserved["hc3_row_id"].duplicated().any():
        duplicate_ids = (
            preserved.loc[preserved["hc3_row_id"].duplicated(), "hc3_row_id"]
            .head(10)
            .tolist()
        )
        raise ValueError(f"Preserved sample has duplicate hc3_row_id values: {duplicate_ids}")
    return preserved[OUTPUT_COLUMNS]


def sample_per_domain(df: pd.DataFrame, samples_per_domain: int, seed: int) -> pd.DataFrame:
    domain_counts = df["source"].value_counts().sort_index()

    if samples_per_domain <= 0:
        raise ValueError("samples-per-domain must be at least 1")

    too_small = domain_counts[domain_counts < samples_per_domain]
    if not too_small.empty:
        details = ", ".join(f"{source}: {count}" for source, count in too_small.items())
        raise ValueError(
            "Not enough rows to sample the requested amount from each domain: "
            f"{details}"
        )

    sampled_df = (
        df.groupby("source", group_keys=False)
        .sample(n=samples_per_domain, random_state=seed)
        .reset_index(drop=True)
    )
    sampled_df = sampled_df.sort_values(["source", "hc3_row_id"]).reset_index(drop=True)
    return serialize_list_columns(sampled_df)


def sample_total(
    df: pd.DataFrame,
    total_samples: int,
    seed: int,
    preserve_path: Optional[Path],
) -> pd.DataFrame:
    if total_samples <= 0:
        raise ValueError("total-samples must be at least 1")
    if total_samples > len(df):
        raise ValueError(
            f"total-samples cannot exceed the dataset size ({len(df)}) without replacement"
        )

    if preserve_path is None:
        sampled_df = df.sample(n=total_samples, random_state=seed).reset_index(drop=True)
        sampled_df = serialize_list_columns(sampled_df)
    else:
        preserved_df = read_preserved_sample(preserve_path)
        if len(preserved_df) > total_samples:
            raise ValueError(
                "Preserved sample has more rows than requested total-samples: "
                f"{len(preserved_df)} > {total_samples}"
            )

        valid_ids = set(df["hc3_row_id"])
        preserved_ids = set(preserved_df["hc3_row_id"])
        missing_ids = sorted(preserved_ids.difference(valid_ids))
        if missing_ids:
            raise ValueError(
                "Preserved sample contains hc3_row_id values that are not in HC3: "
                f"{missing_ids[:10]}"
            )

        remaining_slots = total_samples - len(preserved_df)
        remaining_df = df[~df["hc3_row_id"].isin(preserved_ids)]
        additional_df = remaining_df.sample(
            n=remaining_slots,
            random_state=seed,
        ).reset_index(drop=True)
        additional_df = serialize_list_columns(additional_df)
        sampled_df = pd.concat(
            [preserved_df, additional_df[OUTPUT_COLUMNS]],
            ignore_index=True,
        )

    return sampled_df.sort_values(["source", "hc3_row_id"]).reset_index(drop=True)


def write_outputs(
    sampled_df: pd.DataFrame,
    output_dir: Path,
    seed: int,
    per_domain_suffix: Optional[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = len(sampled_df)
    unified_suffix = f"{total_samples}_seed{seed}"
    unified_path = output_dir / f"hc3_unified_{unified_suffix}.csv"
    sampled_df.to_csv(unified_path, index=False)

    for source, source_df in sampled_df.groupby("source", sort=True):
        source_suffix = per_domain_suffix or f"{len(source_df)}_seed{seed}"
        source_path = output_dir / f"hc3_{source}_{source_suffix}.csv"
        source_df.to_csv(source_path, index=False)

    distribution = sampled_df["source"].value_counts().sort_index()
    print(f"Wrote {len(sampled_df)} sampled rows to {unified_path}")
    print("Sample distribution by source:")
    for source, count in distribution.items():
        print(f"  {source}: {count}")


def main() -> None:
    args = parse_args()
    df = load_hc3()

    if args.total_samples is None:
        if args.preserve_unified_sample is not None:
            raise ValueError("--preserve-unified-sample requires --total-samples")
        sampled_df = sample_per_domain(df, args.samples_per_domain, args.seed)
        per_domain_suffix = f"{args.samples_per_domain}_seed{args.seed}"
    else:
        sampled_df = sample_total(
            df,
            args.total_samples,
            args.seed,
            args.preserve_unified_sample,
        )
        per_domain_suffix = None

    write_outputs(sampled_df, args.output_dir, args.seed, per_domain_suffix)


if __name__ == "__main__":
    main()
