# CSCI544 Final Project

This repo contains the dataset generation and detector evaluation workflows used in the project:

- recursive paraphrase dataset generation
- ZeroGPT-style baseline and GPTZero-like detector evaluation in Colab
- T5 word-perturbation dataset creation

## Recursive paraphrase

The recursive paraphrase pipeline lives in [`HC3-Recursive-Paraphrase`](HC3-Recursive-Paraphrase/README.md).

- Install the folder requirements.
- Add `OPENAI_API_KEY` to `HC3-Recursive-Paraphrase/.env`.
- Run `paraphrase_pipeline.py estimate` or `paraphrase_pipeline.py run`.
- For the full tracked 10k export, use the detailed instructions in the folder README.

See [`HC3-Recursive-Paraphrase/README.md`](HC3-Recursive-Paraphrase/README.md) for the full setup, commands, and output layout.

## ZeroGPT Colab workflow

The evaluator-facing Colab bundle lives in [`ZeroGPT/colab_hc3_bundle`](ZeroGPT/colab_hc3_bundle/README.md).

- Upload the folder to Google Drive.
- Open `ZeroGPT/colab_hc3_bundle/hc3_colab_workflow.ipynb` in Colab.
- Set `REPO_DIR` to the Drive path for the uploaded folder.
- Run the notebook cells in order to retrain/reuse baselines, train/reuse the GPTZero-like detector, and score the kept test datasets.

See [`ZeroGPT/colab_hc3_bundle/README.md`](ZeroGPT/colab_hc3_bundle/README.md) for the full Colab workflow and kept artifact layout.

## Perturbation Dataset Creator

The perturbation workflow is documented only here. No separate README is provided for it.

### File

- `pertubation_dataset_creator.ipynb`

### Purpose

- Load an HC3 unified CSV.
- Perturb only the AI answers with T5 word-mask fills.
- Write a perturbed CSV plus a JSON report.

### Expected input

The notebook expects an HC3 unified CSV with these columns:

- `hc3_row_id`
- `source`
- `question`
- `human_answers`
- `chatgpt_answers`

The default input filename is:

```text
hc3_unified_10000_seed42_clean.csv
```

### How to run

1. Open `pertubation_dataset_creator.ipynb` in Colab.
2. Run the install cell.
3. Upload the HC3 CSV if it is not already in the Colab working directory.
4. Adjust the config values in the notebook if needed:
   `MODEL_NAME`, `PERTURBATIONS_PER_ANSWER`, `GENERATION_BATCH_SIZE`, `MAX_ROWS`, `CHECKPOINT_EVERY`.
5. Run the remaining cells in order.

### Outputs

The notebook writes:

- `hc3_unified_t5_perturbed_ai_clean.csv`
- `hc3_unified_t5_perturbed_ai_clean_report.json`

Only the AI answers are perturbed. Human answers are passed through and reserialized into the same HC3-compatible column shape.
