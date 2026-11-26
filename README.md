# MBTI Reddit 댓글 전처리

Hugging Face의 [minhaozhang/mbti](https://huggingface.co/datasets/minhaozhang/mbti) 원본을 댓글 단위로 정제한 프로젝트입니다. 공개 파일인 **classifier_strict.parquet에는 댓글 9,128,747개(약 1.08 GB)**가 들어 있습니다.

[전처리 설명 웹페이지](https://seahyunjin.github.io/mbti-project/preprocessing.html) · [Parquet 다운로드](https://github.com/seahyunjin/mbti-project/releases/download/data-v1/classifier_strict.parquet) · [데이터 릴리스](https://github.com/seahyunjin/mbti-project/releases/tag/data-v1) · [집계 및 SHA-256](docs/data_manifest.json)

## 원본에서 어떻게 전처리했나

원본 train/test를 합쳐 **1,269,825행**을 처리했습니다. 원본 한 행에 여러 댓글이 묶여 있어서 분리 후에는 **14,017,296개 댓글**이 나왔습니다.

1. **작성자와 라벨 확인.** 작성자에게 MBTI 유형이 둘 이상 있거나 MBTI와 세부 축(E/I, N/S, T/F, J/P)이 일치하지 않으면 그 작성자를 제외하도록 했습니다. 이번 처리에서는 해당 작성자가 0명이었습니다. 작성자 이름을 `U00001` 같은 가명 ID로 바꿨습니다.
2. **사용자 단위로 다시 분할.** 원본 train과 test에 모두 등장한 작성자가 **9,008명**이었습니다. 같은 작성자의 댓글이 여러 분할에 들어가지 않도록, MBTI별로 층화해 사용자를 약 **70:15:15** 비율로 배정했습니다. 결과의 `split` 열에 새 `train`, `val`, `test`가 들어 있습니다.
3. **댓글 분리와 텍스트 정리.** 원본 `body`에서 큰따옴표-세미콜론-큰따옴표 경계(`";"`)로 댓글을 나눴습니다. HTML 문자와 유니코드를 정리하고, 인용 줄·이모트·일부 마크다운 기호를 제거했습니다. 링크의 표시 문구는 남기고 URL·사용자 언급·이메일은 각각 `urltoken`·`usertoken`·`emailtoken`으로 바꿨습니다.
4. **잡음과 비영어 댓글 제외.** 정리 후 비어 있는 댓글, 정해진 봇·운영 문구, 영문 단어 정규식 기준 5단어 미만 댓글을 제외했습니다. 20단어 이상은 Lingua로 판별해 최고 신뢰도가 0.70 이상인 비영어 댓글을 제외했습니다. 20단어 미만과 낮은 신뢰도의 댓글은 `uncertain`으로 보존했습니다.
5. **정확 중복 제거.** 정제 텍스트를 대소문자 구분 없이 비교해 같은 작성자의 중복은 한 건만 남겼습니다. 서로 다른 작성자에게서 발견된 동일한 정제 텍스트는 첫 등장분까지 제거했습니다.
6. **MBTI 라벨 누수 줄이기.** 유형명, 일부 X 표기, 인지기능, MBTI 게시판과 성격 유형론 용어 등 강한 누수 패턴을 찾았습니다. 기본 정제 후 **9,331,486개** 중 강한 누수로 표시된 **202,739개**를 제외해 공개 파일을 만들었습니다. `introvert` 같은 약한 표현은 `leak_weak`로 표시해 남겼습니다.

| 새 분할 | 배정된 사용자 | 공개 파일의 댓글 |
| --- | ---: | ---: |
| train | 7,532 | 6,396,175 |
| val | 1,614 | 1,221,237 |
| test | 1,615 | 1,511,335 |
| 합계 | 10,761 | 9,128,747 |

분할 비율은 사용자 기준입니다. 공개 파일에 댓글이 남은 사용자는 10,757명입니다. 최종 제외 사유와 MBTI별 건수는 [데이터 집계](docs/data_manifest.json)에서 확인할 수 있습니다.

## 현재 남은 문제

| 항목 | 확인한 내용 |
| --- | --- |
| 유형 불균형 | INFP 2,047,104개, ESFP 16,657개로 약 123배 차이입니다. MBTI별 동일 개수 추출은 아직 적용하지 않았습니다. |
| 작성자별 편중 | 작성자당 최대 5개 제한을 적용하지 않았습니다. 댓글이 남은 작성자 중 10,634명이 5개를 넘고, 최댓값은 69,276개입니다. |
| 언어 판정 | `uncertain` 댓글이 4,065,373개(44.5%) 남아 있습니다. 영어로 분류된 댓글만 분석하려면 `language`가 `english`인 행을 선택하세요. |
| 라벨 누수 | `leak_weak=True` 댓글이 14,956개 남아 있습니다. 정규식에 포함되지 않은 간접 표현, 기질 코드, 소시오닉스 코드나 표기 변형도 남을 수 있습니다. |
| 분리·중복 정확도 | 댓글 경계는 원본의 구분자에 의존합니다. 중복 제거는 정제 후 동일한 문자열을 기준으로 하므로 비슷한 복붙이나 의역은 남을 수 있습니다. |
| 가명화와 원본 추적 | 작성자 이름 열은 가명 ID로 대체했습니다. 본문에 개인정보가 남을 수 있습니다. 공개 파일은 `source_split`을 생략하므로 `source_row_id`와 `segment_id`만으로 원본 행을 고유하게 찾을 수 없습니다. |
| 집계 차이 | 여러 작성자 사이의 중복을 최종 제거한 수는 `audit.parquet` 기준 **8,387개**입니다. 기존 `report.json`의 **6,124개**는 첫 등장분을 다시 검사하기 전 수치입니다. |

## 개인 노트북에서 Parquet 읽기

Python 환경에서 pandas와 PyArrow를 설치합니다. Jupyter나 VS Code 노트북을 쓰는 경우, 사용 중인 커널에서 `%pip install pandas pyarrow`를 실행해도 됩니다.

```bash
python -m pip install pandas pyarrow
```

**다운로드한 파일 읽기.** [Parquet 파일](https://github.com/seahyunjin/mbti-project/releases/download/data-v1/classifier_strict.parquet)을 다운로드한 뒤, 노트북의 코드 셀에서 아래를 실행합니다. 파일을 노트북과 같은 폴더에 뒀다면 `path = Path("classifier_strict.parquet")`로 바꾸면 됩니다.

```python
from pathlib import Path

import pandas as pd

path = Path.home() / "Downloads" / "classifier_strict.parquet"

# 댓글 텍스트 없이 분할별·유형별 행 수 확인
df = pd.read_parquet(path, engine="pyarrow", columns=["user_id", "mbti", "split"])
print(f"댓글 수: {len(df):,}")
print(pd.crosstab(df["mbti"], df["split"]))
```

**특정 분할의 댓글 텍스트 읽기.** 다음 예제는 val 댓글과 필요한 열만 읽습니다. 학습용은 `"val"`을 `"train"`으로, 최종 평가용은 `"test"`로 바꿉니다.

```python
val = pd.read_parquet(
    path,
    engine="pyarrow",
    columns=["user_id", "mbti", "text"],
    filters=[("split", "=", "val")],
)
print(val.head(3))
```

**GitHub 주소로 바로 읽기.** 아래 예제는 수동으로 파일을 저장하는 단계를 생략합니다.

```python
import pandas as pd

url = "https://github.com/seahyunjin/mbti-project/releases/download/data-v1/classifier_strict.parquet"
df = pd.read_parquet(url, engine="pyarrow", columns=["user_id", "mbti", "split"])
```

URL로 읽으면 HTTP 읽기 방식에 따라 약 1.08 GB 전체가 전송될 수 있습니다. 같은 파일을 여러 번 분석할 때는 한 번 내려받아 로컬 경로로 읽으세요. `text` 열을 포함하면 압축 파일 크기보다 더 많은 RAM을 쓸 수 있습니다.

## 전처리를 다시 실행하려면

원본 파일을 저장소 아래 `data_hf/train.parquet`와 `data_hf/test.parquet`에 둡니다. [전처리 코드](mbti_full_preprocess.py)는 원문을 배치로 읽고 결과를 `full_preprocessed/`에 저장합니다.

```bash
python -m pip install -r requirements.txt
python -c "from mbti_full_preprocess import preprocess; preprocess()"
```

| 생성 파일 | 내용 |
| --- | --- |
| `classifier_strict.parquet` | 강한 누수 패턴으로 표시된 댓글을 추가로 제외한 분류용 데이터 |
| `classifier_base.parquet` | 강한 누수 제외 전 기본 정제 결과 |
| `rag_train.parquet` | strict 결과에서 train 분할만 저장한 파일 |
| `users.parquet` | 가명 사용자 ID, MBTI, 새 분할 |
| `audit.parquet` · `report.json` | 댓글별 제외 사유와 처리 집계 |

[EDA 노트북](mbti_preprocessed_eda.ipynb)은 로컬에서 생성한 전체 결과를 사용합니다. 공개 Parquet만 받은 경우에는 위의 `pd.read_parquet()` 예제로 바로 확인할 수 있습니다.
