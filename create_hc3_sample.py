#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


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
    return df[["hc3_row_id", "source", "question", "human_answers", "chatgpt_answers"]]


def serialize_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ["human_answers", "chatgpt_answers"]:
        out[column] = out[column].apply(
            lambda value: json.dumps(value, ensure_ascii=False)
        )
    return out


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_hc3()
    domain_counts = df["source"].value_counts().sort_index()

    if args.samples_per_domain <= 0:
        raise ValueError("samples-per-domain must be at least 1")

    too_small = domain_counts[domain_counts < args.samples_per_domain]
    if not too_small.empty:
        details = ", ".join(f"{source}: {count}" for source, count in too_small.items())
        raise ValueError(
            "Not enough rows to sample the requested amount from each domain: "
            f"{details}"
        )

    sampled_df = (
        df.groupby("source", group_keys=False)
        .sample(n=args.samples_per_domain, random_state=args.seed)
        .reset_index(drop=True)
    )
    sampled_df = sampled_df.sort_values(["source", "hc3_row_id"]).reset_index(drop=True)
    sampled_df = serialize_list_columns(sampled_df)

    total_samples = len(sampled_df)
    unified_suffix = f"{total_samples}_seed{args.seed}"
    per_domain_suffix = f"{args.samples_per_domain}_seed{args.seed}"
    unified_path = output_dir / f"hc3_unified_{unified_suffix}.csv"
    sampled_df.to_csv(unified_path, index=False)

    for source, source_df in sampled_df.groupby("source", sort=True):
        source_path = output_dir / f"hc3_{source}_{per_domain_suffix}.csv"
        source_df.to_csv(source_path, index=False)

    distribution = sampled_df["source"].value_counts().sort_index()
    print(f"Wrote {len(sampled_df)} sampled rows to {unified_path}")
    print("Sample distribution by source:")
    for source, count in distribution.items():
        print(f"  {source}: {count}")


if __name__ == "__main__":
    main()
