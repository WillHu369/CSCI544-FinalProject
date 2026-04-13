# HC3 Recursive Paraphrase

This folder builds recursive paraphrase dataset variants from the HC3 sample CSV in the repository. It reuses the HC3 normalization and split logic from `ZeroGPT/colab_hc3_bundle`, but it does not train or score detectors itself. The output is a set of CSV dataset bundles that can be used by the existing ZeroGPT Colab workflow.

## Setup

From this folder, create a virtual environment with a compatible Python version such as Python 3.12:

```bash
python -m venv .venv
```

If your default `python` is not the version you want, use a specific interpreter instead:

```bash
python3.12 -m venv .venv
```

Activate the environment:

```bash
source .venv/Scripts/activate
```

Or in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Store your OpenAI key in a local `.env` file in this folder:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

The script auto-loads that file when it starts.

## Commands

Estimate token usage and projected API cost for the default depth set `1,2,3`:

```bash
python paraphrase_pipeline.py estimate
```

Generate depth `1`, `2`, and `3` dataset variants for a 100-row pilot:

```bash
python paraphrase_pipeline.py run --sample-rows 100 --generator-model gpt-5.4-mini --max-estimated-cost-usd 5
```

Generate a proportional subset with a different model:

```bash
python paraphrase_pipeline.py run --sample-fraction 0.25 --generator-model gpt-5.4-nano
```

Override the depth list if needed:

```bash
python paraphrase_pipeline.py run --depths 2,3
```

## Outputs

Each run writes into `artifacts/experiments/<source>_<subset>_seed<seed>/`:

- `sampled_source_rows.csv`
- `estimate.json`
- `datasets/control/{train,val,test,full}.csv`
- `datasets/<model>/depth_<n>/{train,val,test,full}.csv`
- `checkpoints/<model>_depthmax<n>.jsonl`
- `api_calls/<model>_depthmax<n>.jsonl`
- `generation_manifest.json`

Each dataset directory contains the same split CSV shape expected by the existing ZeroGPT Colab workflow. Use one of the generated dataset directories as the input data folder when you run the training and evaluation workflow in `ZeroGPT/colab_hc3_bundle`.

## Model Selection and Pricing

The CLI accepts arbitrary model IDs through `--generator-model`. Built-in cost estimation is included for the GPT-5.4 family using official OpenAI pricing checked on **April 13, 2026**:

- `gpt-5.4`
- `gpt-5.4-mini`
- `gpt-5.4-nano`

For another model, pass a pricing override:

```bash
python paraphrase_pipeline.py estimate --generator-model my-model --model-pricing my-model=0.75,4.50
```

The override format is `MODEL=INPUT_COST_PER_1M,OUTPUT_COST_PER_1M`.
