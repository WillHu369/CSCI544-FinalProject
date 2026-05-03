# CSCI544 Final Project

This repo contains the dataset generation and detector evaluation workflows used in the project:

- recursive paraphrase dataset generation
- stylistic cleanup (avoidance technique) dataset creation 
- ZeroGPT-style baseline and GPTZero-like detector evaluation in Colab
- T5 word-perturbation dataset creation
- Binoculars evaluation notes and artifacts
- DetectGPT detector evaluation

## Environment setup

This project uses both a local Python environment and Google Colab.

- `HC3-Recursive-Paraphrase` runs in a local Python environment. Install `HC3-Recursive-Paraphrase/requirements.txt` and add `OPENAI_API_KEY` to `HC3-Recursive-Paraphrase/.env`.
- `Stylistic-Cleanup` runs in a Google Colab. Follow the instructions in the folder README to install the required libraries.Upload the file and the HC3 dataset to Google Drive and run `RB_final_project_stylistic_cleanup.ipynb`. 
- `ZeroGPT/colab_hc3_bundle` runs in Google Colab. Upload the folder to Google Drive and run `hc3_colab_workflow.ipynb`; the notebook handles its runtime installs.
- `pertubation_dataset_creator.ipynb` runs in Google Colab. Open the notebook and run its install cell before the generation cells.
- `Binoculars` reproduction notes are in [`Binoculars/README.md`](Binoculars/README.md).
- `DetectGPT` runs in Google Colab. Upload the folder to Google Drive and run `DetectGPT.ipynb`; the notebook handles its runtime installs. The readme on the instructions is at [`DetectGPT-colab/README.md`](DetectGPT-colab/README.md).

## Device / system used

The code in this repo was run in two environments:

- a local Windows/Python environment for the recursive paraphrase pipeline
- Google Colab GPU runtimes for the ZeroGPT-style baselines, GPTZero-like detector, perturbation notebook, and Binoculars workflow

## How to run

- Recursive paraphrase: follow [`HC3-Recursive-Paraphrase/README.md`](HC3-Recursive-Paraphrase/README.md).
- Stylistic cleanup follow [`Stylistic-Cleanup/README.md`](Stylistic-Cleanup/README.md)
- ZeroGPT Colab workflow: follow [`ZeroGPT/colab_hc3_bundle/README.md`](ZeroGPT/colab_hc3_bundle/README.md).
- Perturbation dataset creator: use the inline instructions below in this README.
- Binoculars: follow [`Binoculars/README.md`](Binoculars/README.md).
- DetectGPT: follow [`DetectGPT-colab/README.md`](DetectGPT-colab/README.md)

## How results are generated

- The recursive paraphrase pipeline reads an HC3 unified CSV and writes control plus recursive-depth paraphrase CSV exports and manifests.
- The stylistic cleanup reads an HC3 unified CSV and perturbs AI data by removing em dashes, emojs, and converting all markdown formatting (lists, headers, bold text, bullet points) to standard/prose text. It writes the perturbed CSV out. 
- The perturbation notebook reads the HC3 CSV, perturbs only the AI answers with T5 mask filling, and writes a perturbed CSV plus a JSON report.
- The ZeroGPT Colab workflow trains or reuses the SVM-TF-IDF, XGBoost-TF-IDF, and GPTZero-like detectors, evaluates the kept test datasets, and writes metric outputs under `artifacts/runs/` and `metrics_share/`.
- The Binoculars workflow scores the kept evaluation sets and produces metric artifacts for comparison with the other detectors.

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

### Perturbation notebook steps

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

## Binoculars Colab workflow

The Binoculars notebook lives in [`Binoculars/Binoculars_Colab.ipynb`](Binoculars/Binoculars_Colab.ipynb) and is designed to run in Google Colab with a GPU runtime.

1. Open `Binoculars/Binoculars_Colab.ipynb` in Colab and select a GPU runtime.
2. Upload `Binoculars/evaluate_samples.py`.
3. Upload one or more HC3 CSVs from [`HC3-Dataset-Samples/`](HC3-Dataset-Samples/) (HC3 "wide" schema).
4. Run the notebook cells in order to score samples, export `summary.json`, and export `roc_curve.csv`.

Tracked metric artifacts (JSON summaries and ROC-point CSVs) are kept under:

```text
Metrics/Binoculars/
```

## DetectGPT Colab workflow

The DetectGPT notebook lives in [`DetectGPT-colab/DetectGPT.ipynb`](DetectGPT-colab/DetectGPT.ipynb) and is designed to run in Google Colab with the project folder stored in Google Drive.

- Upload the full `DetectGPT-colab` folder to Drive, including `DetectGPT.ipynb`, `run.py`, `custom_datasets.py`, `calculate_metrics.py`, `avoidance_run.py`, `requirements.txt`, and `data/`.
- Open `DetectGPT.ipynb` in Colab, install the notebook dependencies, mount Drive, and set `PROJECT_PATH` to the uploaded folder.
- Make sure the HC3 data is available in `data/hc3` before running the detector.
- Run the smoke test cell first, then use the larger run cells and pass the generated `.json` file to `calculate_metrics.py`.
- Results are written under `results/`, and the notebook uses a Drive-backed `hf_cache` for Hugging Face assets.

See [`DetectGPT-colab/README.md`](DetectGPT-colab/README.md) for the full notebook-specific setup and command examples.

### Outputs

The notebook writes:
```results/{results path here}/perturbation_{number of perturbations}_d_results.json```

This file contains the results of the test, which is used by calculate_metrics.py to generate the relevant metrics (F1 score, TPR @0.1 FPR, etc). Like so:

```python calculate_metrics.py --path results/{results path here}/perturbation_20_d_results.json```

The path to the results can usually be seen in the console output log. The code will begin by writing into tmp_results folder first, and once it is finished, it will move to the results folder. The log will include the path.
