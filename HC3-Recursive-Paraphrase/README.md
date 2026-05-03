# HC3 Recursive Paraphrase

This folder generates recursive paraphrase variants of the HC3 unified dataset. It reuses the HC3 normalization and split logic from `ZeroGPT/colab_hc3_bundle`, but it does not train or score detectors itself.

## Environment setup

- Python with the dependencies in `requirements.txt`
- an OpenAI API key
- an HC3 CSV source file

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create `HC3-Recursive-Paraphrase/.env` with:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

The script loads that file automatically.

## Device / system used

- Platform used for the tracked export: local Windows/Python environment
- Shell used for the commands in this repo: PowerShell
- External dependency: OpenAI API access for generation

## Input Files

Default source file:

```text
../HC3-Dataset-Samples/hc3_unified_1000_seed42.csv
```

For the tracked 10k recursive export kept in this repo, use:

```text
../HC3-Dataset-Samples/hc3_unified_10000_seed42_clean.csv
```

Optional prompt prefix file:

```text
prompt_prefix.txt
```

## How to run

Estimate token usage and API cost with the default settings:

```bash
python paraphrase_pipeline.py estimate
```

Estimate the tracked 10k-source run:

```text
python paraphrase_pipeline.py estimate --source-file ..\HC3-Dataset-Samples\hc3_unified_10000_seed42_clean.csv --generator-model gpt-5.4-mini --depths 1,2,3
```

Run the full 10k source file:

```text
python paraphrase_pipeline.py run --source-file ..\HC3-Dataset-Samples\hc3_unified_10000_seed42_clean.csv --generator-model gpt-5.4-mini --depths 1,2,3
```

Useful flags:

- `--sample-rows N`
- `--sample-fraction X`
- `--depths 1,2,3`
- `--quality-check-depth 3`
- `--quality-min-score 3`
- `--max-estimated-cost-usd VALUE`
- `--num-shards N --shard-index K`

## How results are generated

1. `estimate` computes token and cost estimates for the chosen source file, model, and recursion depths.
2. `run` reads the source HC3 CSV, builds the control split, and generates recursive paraphrases for each requested depth.
3. The script writes the generated CSV datasets, checkpoints, API call logs, and manifest files under the output directory for that run.
4. The canonical tracked final export in this repo is `datasets_final_hc3_unified_10000_gpt54mini_depths123/`.

## Output Layout

Per-run artifacts are written under:

```text
artifacts/experiments/<source>_<subset>_seed<seed>/
```

Typical contents:

- `sampled_source_rows.csv`
- `estimate.json`
- `datasets/control/{train,val,test,full}.csv`
- `datasets/<model>/depth_<n>/{train,val,test,full}.csv`
- `checkpoints/<model>_depthmax<n>.jsonl`
- `api_calls/<model>_depthmax<n>.jsonl`
- `generation_manifest.json`
