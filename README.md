# MBTI Reddit comment preprocessing

## 전처리 설명과 데이터

- [전처리 설명 웹페이지](https://seahyunjin.github.io/mbti-project/preprocessing.html): 처리 순서, 사용자 단위 분할, 실제 제외 집계, 현재 한계를 정리했습니다. [HTML 파일](docs/preprocessing.html)도 제공합니다.
- [데이터 manifest](docs/data_manifest.json): 파일 크기, 행 수, 유형별 분포와 SHA-256을 기록했습니다.
- [데이터 Release](https://github.com/seahyunjin/mbti-project/releases/tag/data-v1)에 `classifier_strict.parquet` (9,128,747행, 약 1.08GB)을 공개했습니다. 아래 URL로 읽을 수 있습니다.

```python
import pandas as pd
url = "https://github.com/seahyunjin/mbti-project/releases/download/data-v1/classifier_strict.parquet"
df = pd.read_parquet(url, columns=["user_id", "mbti", "split"])
```

URL 읽기도 네트워크 전송이 발생합니다. 텍스트 전체를 읽으면 압축 파일보다 많은 RAM이 필요합니다. 현재 파일은 유형별 균형 추출과 사용자당 5개 제한을 적용하기 전의 결과입니다.

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
