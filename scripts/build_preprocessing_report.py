"""Build an aggregate-only HTML report and release manifest from existing outputs."""
from collections import Counter
from hashlib import sha256
from html import escape
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "full_preprocessed"
DOCS = ROOT / "docs"
RELEASE_URL = "https://github.com/seahyunjin/mbti-project/releases/download/data-v1/classifier_strict.parquet"


def table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{escape(f'{v:,}' if isinstance(v, int) else str(v))}</td>" for v in row) + "</tr>"
        for row in rows) + "</tbody></table>"


def main():
    DOCS.mkdir(exist_ok=True)
    report = json.loads((DATA / "report.json").read_text())
    users = pd.read_parquet(DATA / "users.parquet")
    path = DATA / "classifier_strict.parquet"
    pf = pq.ParquetFile(path)
    by_type, by_split, languages, authors = [Counter() for _ in range(4)]
    weak = short = 0
    for batch in pf.iter_batches(batch_size=250_000, columns=["user_id", "mbti", "split", "language", "leak_weak", "word_count"]):
        d = batch.to_pandas()
        for counter, col in [(by_type, "mbti"), (by_split, "split"), (languages, "language"), (authors, "user_id")]:
            counter.update({str(k): int(v) for k, v in d[col].value_counts().items()})
        weak += int(d.leak_weak.sum())
        short += int((d.word_count < 5).sum())
    reasons = Counter()
    for batch in pq.ParquetFile(DATA / "audit.parquet").iter_batches(batch_size=250_000, columns=["reason"]):
        reasons.update({str(k): int(v) for k, v in batch.to_pandas().reason.value_counts().items()})
    digest = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            digest.update(chunk)
    manifest = {
        "file": path.name, "release_url": RELEASE_URL, "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(), "rows": pf.metadata.num_rows,
        "columns": pf.schema.names, "source_rows": report["source_rows"],
        "source_train_test_author_overlap": report["source_train_test_author_overlap"],
        "authors_in_users_file": len(users), "authors_with_strict_comments": len(authors),
        "user_splits": {str(k): int(v) for k, v in users.split.value_counts().items()},
        "comment_splits": dict(by_split), "mbti_counts": dict(sorted(by_type.items())),
        "languages": dict(languages), "weak_leakage_comments": weak,
        "under_five_word_comments": short,
        "authors_above_five_comments": sum(v > 5 for v in authors.values()),
        "max_comments_per_author": max(authors.values()),
        "final_audit_reasons": dict(reasons),
        "original_report": report,
        "processing_code_commit": "e377e6d9bd9198da0a08f4b54532d2de2e3b5bc7",
        "limitations": [
            "Unbalanced classes; no five-comments-per-author cap.",
            "Uncertain language and weak leakage are retained.",
            "User IDs are pseudonyms; text is not guaranteed to be free of personal information.",
            "Original report cross-author count precedes second-pass correction; final audit count is authoritative.",
            "Strict output omits source_split, so source_row_id plus segment_id is not globally unique."
        ]
    }
    (DOCS / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    split_table = table(["분할", "배정 사용자", "최종 댓글"], [(s, int((users.split == s).sum()), by_split[s]) for s in ("train", "val", "test")])
    reason_table = table(["최종 audit 사유", "댓글 수"], [(k or "통과: classifier_base", int(v)) for k, v in reasons.most_common()])
    class_table = table(["MBTI", "댓글 수"], [(k, int(v)) for k, v in sorted(by_type.items())])
    schema_table = table(["열", "의미"], [
        ("user_id", "작성자별 가명 ID. U00001 형식"),
        ("mbti", "16개 MBTI 중 제공된 라벨"),
        ("split", "새로 배정한 train / val / test"),
        ("source_row_id", "원본 파일 안에서의 행 번호. 원본 train/test 간 번호가 겹칠 수 있음"),
        ("segment_id", "원본 행 안에서 분리된 댓글의 1부터 시작하는 번호"),
        ("text", "정제된 댓글 텍스트"),
        ("text_hash", "정제 텍스트를 casefold한 뒤 계산한 SHA-256"),
        ("word_count", "영문 단어 정규식으로 센 개수"),
        ("leak_weak", "약한 성격 관련 표현 존재 여부. strict에서도 유지됨"),
        ("language", "english 또는 uncertain")
    ])
    pipeline = [
        ("사용자 통합·분할", "원본 train/test의 작성자·라벨 메타데이터를 합칩니다. 한 작성자에게 여러 MBTI가 있거나 EI·NS·FT·JP 축이 MBTI와 불일치하면 제외합니다. 현재 제외 작성자는 0명입니다. 작성자 이름을 정렬해 U00001 형식의 가명 ID를 만들고, MBTI로 층화한 train_test_split(random_state=42)로 사용자 단위 70:15:15 분할을 배정합니다. 원문 텍스트는 두 파일에서 배치로 읽습니다."),
        ("댓글 분리", '원본 행의 양 끝 큰따옴표를 제거한 뒤 정확히 큰따옴표-세미콜론-큰따옴표(";") 경계를 사용해 댓글을 나눕니다. 모든 세미콜론을 분리하는 방식은 아닙니다. 이 경계는 잠정 규칙이며 의미상 댓글 경계를 전수 검증한 것은 아닙니다.'),
        ("텍스트 정리", "HTML entity 복원 → Unicode NFKC 정규화 → 줄 앞의 > 인용 제거 → 이모트 제거 → Markdown 링크의 표시 글자 보존 및 urltoken 추가 → 이메일을 emailtoken으로 치환 → URL을 urltoken으로 치환 → u/사용자·@사용자 언급을 usertoken으로 치환 → 나머지 링크·HTML 태그·일부 Markdown 기호 정리 → 연속 공백 축약 순서입니다. 인용 뒤 별도 줄에 있는 답글은 남깁니다."),
        ("잡음·언어 판정", "빈 댓글, 미리 정의된 봇·운영자 문구, 영문 단어 정규식 기준 5단어 미만 댓글을 제외합니다. 20단어 미만은 언어 판정을 하지 않고 uncertain으로 남깁니다. 그 이상은 Lingua의 10개 언어 후보 중 최고 신뢰도가 0.70 이상일 때 english 또는 non_english로 판정하며, 그보다 낮으면 uncertain입니다. non_english만 제외합니다."),
        ("정확 중복 제거", "정제 텍스트를 casefold한 SHA-256으로 동일 문장을 찾습니다. 같은 사용자의 중복은 한 건만 남깁니다. 여러 사용자에게 등장한 동일 텍스트는 두 번째 패스에서 처음 등장한 댓글까지 제외합니다. 유사 문장·의역 중복을 탐지하는 절차는 없습니다."),
        ("MBTI 누수와 결과 저장", "완전한 MBTI 유형, 일부 X 유형, 구분기호를 넣은 유형, 코드에 나열된 인지기능 표기, MBTI·인지기능·에니어그램 용어 및 특정 MBTI 게시판 패턴을 high로 표시합니다. classifier_strict와 rag_train은 high가 발견된 댓글 전체를 제외합니다. introvert·thinker·neuroticism 등 weak 표현은 표시만 하며 남깁니다. 소시오닉스 개별 코드나 모든 인지기능 대소문자 변형을 포괄하지는 않습니다.")
    ]
    steps = "".join(f"<article><span class='num'>{i:02}</span><div><h3>{escape(title)}</h3><p>{escape(body)}</p></div></article>" for i, (title, body) in enumerate(pipeline, 1))
    code = escape(f'''import pandas as pd

# GitHub Release에 data-v1 데이터가 공개된 후 사용하는 주소입니다.
url = "{RELEASE_URL}"

# 우선 필요한 열만 읽기
df = pd.read_parquet(url, columns=["user_id", "mbti", "split"])
print(df.shape)
print(pd.crosstab(df["mbti"], df["split"]))

# 텍스트를 포함한 학습용 분할을 읽으려면:
# train = pd.read_parquet(url, filters=[("split", "=", "train")],
#                         columns=["user_id", "mbti", "text"])
''')
    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MBTI 전처리 과정과 데이터 안내</title>
<style>
:root{{--ink:#15302f;--muted:#5c706c;--accent:#087b68;--bg:#f3f5ef;--line:#dce3d8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.8 system-ui,-apple-system,"Apple SD Gothic Neo",sans-serif}}
main{{max-width:1080px;margin:auto;padding:48px 24px 80px}}header{{padding:36px 0 24px;border-bottom:2px solid var(--ink)}}.eyebrow{{font-size:12px;letter-spacing:.15em;color:var(--accent);font-weight:800}}h1{{font-size:clamp(30px,5vw,52px);line-height:1.2;margin:16px 0}}h2{{font-size:26px;margin:48px 0 18px}}h3{{margin:0;font-size:20px}}p{{margin:10px 0}}a{{color:var(--accent)}}nav{{display:flex;gap:18px;flex-wrap:wrap;margin:22px 0}}.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0}}.stat{{background:white;border:1px solid var(--line);padding:22px;border-radius:12px}}.stat strong{{font-size:29px;display:block}}.stat span,.muted{{color:var(--muted)}}.note{{background:#fff4d7;padding:20px;border-left:4px solid #c18a14;border-radius:4px}}article{{display:flex;gap:24px;padding:24px 0;border-bottom:1px solid var(--line)}}.num{{font-size:28px;color:var(--accent);font-weight:800;min-width:40px}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:12px 16px;text-align:left;border-bottom:1px solid var(--line)}}th{{background:#e4ece3}}.scroll{{overflow-x:auto}}pre{{overflow:auto;background:#142c2a;color:#e2f6e9;padding:24px;border-radius:12px;font-size:14px;line-height:1.65}}code{{font-family:ui-monospace,monospace}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}footer{{margin-top:40px;padding-top:20px;border-top:1px solid var(--line);font-size:14px}}@media(max-width:700px){{.stats,.two{{grid-template-columns:1fr}}main{{padding:24px 16px}}}}
</style></head><body><main>
<header><div class="eyebrow">MBTI PROJECT / PREPROCESSING RECORD</div><h1>댓글 데이터는<br>어떻게 정리했나요?</h1>
<p>실제 전처리 코드와 저장된 Parquet 집계를 바탕으로 작성한 데이터 처리 기록입니다. 원본 행, 분리된 댓글, 모델 입력용 댓글을 구분합니다.</p>
<p class="muted">원본: <a href="https://huggingface.co/datasets/minhaozhang/mbti">minhaozhang/mbti</a> · 결과 검토: 2026-11-26</p></header>
<nav><a href="#pipeline">처리 순서</a><a href="#split">데이터 분할</a><a href="#counts">제외 내역</a><a href="#limits">현재 한계</a><a href="#read">파일 읽기</a></nav>
<div class="stats"><div class="stat"><span>원본 행</span><strong>{report['source_rows']:,}</strong></div><div class="stat"><span>분리된 댓글</span><strong>{sum(reasons.values()):,}</strong></div><div class="stat"><span>엄격 분류용 댓글</span><strong>{pf.metadata.num_rows:,}</strong></div></div>
<div class="note"><strong>현재 파일은 기본 정제 결과입니다.</strong><p>유형별 동일 개수 추출과 작성자당 최대 5개 제한은 아직 적용되지 않았습니다. 영어가 확실하지 않은 댓글과 약한 성격 관련 표현도 일부 남아 있습니다.</p></div>
<section id="pipeline"><h2>01 · 전처리 순서</h2>{steps}<p class="muted">검토한 코드: <a href="https://github.com/seahyunjin/mbti-project/blob/e377e6d9bd9198da0a08f4b54532d2de2e3b5bc7/mbti_full_preprocess.py">mbti_full_preprocess.py</a>. 학습 데이터에는 원래 작성자 이름 열을 저장하지 않지만, 댓글 본문에서 모든 개인정보가 제거됐다는 보장은 없습니다.</p></section>
<section id="split"><h2>02 · train/test를 다시 나눈 이유</h2><p>원본 train과 test 양쪽에 등장한 작성자가 {report['source_train_test_author_overlap']:,}명이었습니다. 같은 작성자를 통해 평가가 쉬워지는 문제를 줄이기 위해 기존 분할을 함께 고려한 뒤 새 분할을 사용자 단위로 배정했습니다. 같은 사용자의 댓글은 한 분할에만 속합니다.</p>{split_table}
<p>배정 사용자 {len(users):,}명 중 최종 strict 댓글이 남은 사용자는 {len(authors):,}명입니다. 70:15:15는 사용자 비율이며, 사용자별 활동량이 달라 댓글 비율은 다릅니다. 데이터가 추가되면 현재 정렬·층화 분할의 ID와 배정도 달라질 수 있습니다.</p></section>
<section id="counts"><h2>03 · 실제 저장 파일의 집계</h2><div class="two"><div><h3>최종 audit 사유</h3>{reason_table}</div><div><h3>Strict 유형별 댓글</h3>{class_table}</div></div>
<p>기본 정제 {reasons['']:,}건 중 high 누수로 표시된 {reasons['']-pf.metadata.num_rows:,}건을 추가로 제외해 strict 파일을 만들었습니다. RAG 후보는 strict의 train 분할 {by_split['train']:,}건입니다.</p>
<p class="note">기존 report.json의 cross_author_duplicate는 1차 집계 {report['counts']['cross_author_duplicate']:,}건입니다. 2차 패스에서 첫 등장 댓글도 제거한 최종 audit 집계는 {reasons['cross_author_duplicate']:,}건입니다. 이 문서에는 최종 audit 값을 사용했습니다.</p></section>
<section id="limits"><h2>04 · 사용 전 알아둘 한계</h2><ul>
<li>언어 판정 uncertain: {languages['uncertain']:,}건 ({languages['uncertain']/pf.metadata.num_rows:.1%}). 이 항목이 모두 비영어라는 뜻도, 영어로 검증됐다는 뜻도 아닙니다.</li>
<li>Strict에도 weak 누수 표시 댓글 {weak:,}건이 남습니다. 유형론 관련 표현을 완전히 제거한 파일은 아닙니다.</li>
<li>5단어 미만 보존 댓글은 {short:,}건입니다. 단어 수는 영문 정규식 기준이며 urltoken 등 치환 토큰도 셉니다.</li>
<li>댓글이 5개를 초과하는 작성자는 {manifest['authors_above_five_comments']:,}명이며 한 작성자의 최대 댓글 수는 {manifest['max_comments_per_author']:,}개입니다.</li>
<li>동일 문자열 중복을 제거합니다. 유사 복붙, 인용 경계 오판, 문맥상 누수까지 전수 검증한 것은 아닙니다.</li>
<li>원본 split을 합쳐 중복 텍스트를 제거했으므로 test의 존재가 코퍼스 필터링에 영향을 줍니다. 독립 배포 평가를 주장하려면 데이터 구성 정책을 따로 검토해야 합니다.</li>
<li>user_id는 가명화입니다. 작성자 자체 진술 라벨과 본문 개인정보의 정확성·완전한 비식별성을 보장하지 않습니다.</li>
</ul></section>
<section id="read"><h2>05 · 공유 파일과 읽는 방법</h2><p><code>classifier_strict.parquet</code> · {manifest['bytes']:,} bytes · {manifest['bytes']/1024**3:.2f} GiB</p>
<p><a href="{RELEASE_URL}">Parquet 다운로드</a> · <a href="https://github.com/seahyunjin/mbti-project/releases/tag/data-v1">데이터 Release</a> · <a href="data_manifest.json">행 수·SHA-256 확인</a></p>
<p class="muted">다운로드 링크는 data-v1 Release 게시 후 동작합니다. URL 읽기도 실행 환경으로 데이터를 전송하며, HTTP 구현에 따라 파일 전체를 받을 수 있습니다. 전체 텍스트를 DataFrame으로 읽으면 압축 파일보다 많은 메모리가 필요합니다.</p>
<pre><code>{code}</code></pre><div class="scroll">{schema_table}</div>
<p>split으로 필터링할 때 val도 별도로 유지하세요. source_row_id는 각 원본 파일에서 0부터 시작하고 strict에는 source_split이 없으므로, source_row_id와 segment_id만으로 원본 댓글을 전역 식별하면 안 됩니다. 정확한 추적에는 audit의 source_split도 필요합니다.</p></section>
<footer>집계 생성: <code>python scripts/build_preprocessing_report.py</code><br>이 문서는 실제 댓글 예시나 원본 작성자 이름을 포함하지 않습니다. 파이프라인의 현재 상태를 설명하며 추가 전처리를 수행하지 않습니다.</footer>
</main></body></html>"""
    (DOCS / "preprocessing.html").write_text(document, encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["rows", "bytes", "sha256", "weak_leakage_comments"]}, indent=2))


if __name__ == "__main__":
    main()

