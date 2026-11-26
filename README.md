# MBTI Reddit comment preprocessing

This repository contains the preprocessing code and exploratory analysis notebook for an MBTI comment classification project. The raw and processed Parquet datasets are not stored in Git.

## Files

- `mbti_full_preprocess.py`: stream the original train/test Parquet files, split bundled comments, clean text, remove duplicates, and write audit and model-ready outputs.
- `mbti_preprocessed_eda.ipynb`: basic EDA of processed outputs. It requires local processed files and currently contains code, not saved EDA output.
- `requirements.txt`: Python dependencies.

## Data layout

Put the original files at `data_hf/train.parquet` and `data_hf/test.parquet`. The preprocessing script writes results to `full_preprocessed/` by default:

| File | Purpose |
| --- | --- |
| `classifier_strict.parquet` | Cleaned comments for classification, excluding marked high-risk MBTI leakage |
| `classifier_base.parquet` | Cleaned comments before high-risk leakage exclusion |
| `rag_train.parquet` | Strict comments from training users |
| `users.parquet` | Anonymous user IDs, MBTI labels, and user-level splits |
| `audit.parquet` | All segmented comments, including exclusion reasons |
| `report.json` | Preprocessing counts |

The source train/test split is not reused because authors occur in both source files. The script assigns new, stratified train/validation/test splits by author and stores the split in each output row.

## Current data limitations

The current outputs have **not** been balanced to equal MBTI class sizes, and the proposed five-comment-per-author cap has **not** been applied. The strict output excludes marked high-risk leakage, but weak personality terms can remain; uncertain language predictions are retained. These decisions should be reviewed before model training.

## Run

Install dependencies with `python -m pip install -r requirements.txt`. Run `python -c "from mbti_full_preprocess import preprocess; preprocess()"` from this directory, then open `mbti_preprocessed_eda.ipynb` to inspect the outputs. The EDA notebook reads local files under `full_preprocessed/`.

For remote EDA without a manual file download, the dataset must first be hosted at a public or authorized URL. A GitHub code repository alone does not serve the locally stored Parquet files.
