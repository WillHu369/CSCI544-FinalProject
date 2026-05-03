# Stylistic Cleanup

Applies rule-based stylistic normalization to the `chatgpt_answers` column of the HC3 dataset to reduce surface-level stylistic markers that tend to distinguish AI-generated text from human text.

## What It Does

Three cleanup passes are applied to each answer:

1. **Em dash removal** — replaces em dashes (`—`, `–`) with commas, handling spaced and unspaced versions 
2. **Emoji removal** — strips all emoji characters using regex 
3. **Markdown formatting removal** — converts headers, bold, bullet points, and numbered lists to plain prose

Each row is flagged with `stylistic_changed: True/False` to track whether any modification was made.

## Output Columns

| Column | Description |
|---|---|
| `chatgpt_answers_stylistic` | Cleaned version of `chatgpt_answers` |
| `stylistic_changed` | Whether any cleanup was applied to the row |

## How to Run

1. Open `RB_final_project_stylistic_cleanup.ipynb`
2. Mount Google Drive & Ensure the HC3 dataset is at:
   ```
   /content/drive/MyDrive/colab_hc3_bundle/HC3-Dataset-Samples/hc3_unified_10000_seed42.csv
   ```
3. Run all cells in order
4. Output CSV is saved to:
   ```
   Stylistic-Cleanup/hc3_sample_10K_stylistic_cleanup.csv
   ```

## Dependencies

```
pip install emoji regex datasets pandas
```

## Results

Of the 10K sample, only ~34% of the rows were modified after the stylistic changes (meaning 2/3 of rows did not have obvious stylistic AI markers) 
