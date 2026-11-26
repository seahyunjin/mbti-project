"""Full-corpus, auditable preprocessing for MBTI classification and later RAG.

The notebook mbti_EDA.ipynb is the interactive entry point. Each source row is
a bundle of comments; a literal '";"' is the provisional boundary. Raw text
never appears in the public processed output. No balancing or author cap is
applied here. Split is assigned by author, independently of source split.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
TYPES = {a+b+c+d for a in 'IE' for b in 'NS' for c in 'TF' for d in 'JP'}
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)*", re.I)
BOUNDARY = re.compile(r'";"')
QUOTE_LINE = re.compile(r'^\s*(?:>+|&gt;+)\s?')
MD_LINK = re.compile(r'\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)', re.I)
EMOTE = re.compile(r'\[\]\(/[^)]+\)', re.I)
URL = re.compile(r'(?:(?:https?://)|(?:www\.))[^\s<>]+', re.I)
MENTION = re.compile(r'(?<!\w)(?:/?u/|@)[A-Za-z0-9_-]+', re.I)
TYPE = re.compile(r"(?<![A-Za-z])[IE][NS][TF][JP](?:['’]?s)?(?:-[AT])?(?![A-Za-z])", re.I)
PARTIAL = re.compile(r'(?<![A-Za-z])(?:[IE][NSX][TFX][JPX]|X[NS][TF][JP])(?![A-Za-z])', re.I)
TYPE_SUB = re.compile(r'(?<![A-Za-z])[IE][ .-]+[NS][ .-]+[TF][ .-]+[JP](?![A-Za-z])', re.I)
FUNC = re.compile(r'(?<![A-Za-z])(?:Ni|Ne|Si|Se|Ti|Te|Fi|Fe)(?:-dom)?(?![A-Za-z])')
STRONG = re.compile(r'\b(?:mbti|myers[ -]*briggs|socionics|enneagram|typology|cognitive functions?|function stacks?|\d+w\d|sx/so|so/sx|sp/so|so/sp|sx/sp|sp/sx)\b', re.I)
SUBREDDIT = re.compile(r'(?i)(?:/?r/)(?:mbti|(?:[ie][ns][tf][jp])(?:memes)?)(?![a-z])')
WEAK = re.compile(r'\b(?:introverts?|extroverts?|extraverts?|intuitions?|thinkers?|feelers?|doorslam|openness|neuroticism|agreeableness|conscientiousness)\b', re.I)
BOT = re.compile(r'(?i)(?:your (?:submission|post|comment) has been removed|i am a bot|this action was performed automatically|remindme!|please contact the moderators)')
EMAIL = re.compile(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b')


def split_comments(body: str) -> tuple[list[str], bool]:
    """Provisional source boundary; report suspicious edge quoting."""
    body = str(body or '')
    suspicious = not (body.startswith('"') and body.endswith('"'))
    if body.startswith('"'):
        body = body[1:]
    if body.endswith('"'):
        body = body[:-1]
    return BOUNDARY.split(body), suspicious


def remove_quotes(text: str) -> str:
    """Remove Reddit Markdown quote lines, preserving following reply lines."""
    lines = text.splitlines()
    kept = [line for line in lines if not QUOTE_LINE.match(line)]
    return '\n'.join(kept)


def clean_comment(raw: str) -> tuple[str, str, int]:
    text = unicodedata.normalize('NFKC', html.unescape(raw)).replace('’', "'")
    text = remove_quotes(text)
    text = EMOTE.sub(' ', text)
    text = MD_LINK.sub(lambda m: m.group(1) + ' urltoken', text)
    text = EMAIL.sub(' emailtoken ', text)
    text = URL.sub(' urltoken ', text)
    text = MENTION.sub(' usertoken ', text)
    # Keep words, apostrophes, punctuation, and negation; remove markup only.
    text = re.sub(r'\[(.*?)\]\((?!urltoken)[^)]*\)', r'\1', text)
    text = re.sub(r'</?[^>\n]+>', ' ', text)
    text = re.sub(r'(?<!\w)[*_~^`]+|[*_~^`]+(?!\w)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = TOKEN.findall(text)
    if not text:
        reason = 'empty_after_quotes'
    elif BOT.search(text):
        reason = 'bot_or_moderation'
    elif len(words) < 5:
        reason = 'under_5_words'
    else:
        reason = ''
    return text, reason, len(words)


def leakage_flags(text: str) -> tuple[bool, bool]:
    high = any(r.search(text) for r in [TYPE, PARTIAL, TYPE_SUB, FUNC, STRONG, SUBREDDIT])
    weak = bool(WEAK.search(text))
    return high, weak


def source_metadata(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Read only author/label columns; do not load all text into RAM."""
    frames = []
    for path in paths:
        d = pd.read_parquet(path, columns=['author', 'mbti', 'EI', 'NS', 'FT', 'JP'])
        d.insert(0, 'source_row_id', np.arange(len(d), dtype=np.int64))
        d['source_split'] = path.stem
        frames.append(d)
    meta = pd.concat(frames, ignore_index=True)
    consistent_axes = (meta.mbti.str[0].eq(meta.EI) & meta.mbti.str[1].eq(meta.NS)
                       & meta.mbti.str[2].eq(meta.FT) & meta.mbti.str[3].eq(meta.JP))
    bad_labels = set(meta.loc[~consistent_axes | ~meta.mbti.isin(TYPES), 'author'])
    conflicts = set(meta.groupby('author').mbti.nunique().loc[lambda s:s.gt(1)].index) | bad_labels
    authors = meta.loc[~meta.author.isin(conflicts), ['author','mbti']].drop_duplicates()
    # Source splits are ignored. Hashing makes the author split stable as rows grow.
    authors = authors.sort_values('author').reset_index(drop=True)
    authors['user_id'] = [f'U{i:05d}' for i in range(1, len(authors)+1)]
    train, held = train_test_split(authors, test_size=.30, stratify=authors.mbti, random_state=42)
    val, test = train_test_split(held, test_size=.50, stratify=held.mbti, random_state=42)
    authors = pd.concat([train.assign(split='train'), val.assign(split='val'),
                         test.assign(split='test')], ignore_index=True)
    return meta, authors, conflicts


def detect_english(text: str, detector, min_words=20, confidence=.70) -> str:
    """Return english/non_english/uncertain; short text is not force-labeled."""
    if len(TOKEN.findall(text)) < min_words:
        return 'uncertain'
    if detector is None:
        return 'not_checked'
    candidates = detector.compute_language_confidence_values(text)
    if not candidates:
        return 'uncertain'
    best = candidates[0]
    return 'english' if best.language.name == 'ENGLISH' and best.value >= confidence else 'non_english' if best.value >= confidence else 'uncertain'


def make_detector():
    from lingua import Language, LanguageDetectorBuilder
    return LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.KOREAN, Language.SPANISH,
        Language.FRENCH, Language.GERMAN, Language.PORTUGUESE, Language.ITALIAN,
        Language.RUSSIAN, Language.SERBIAN, Language.DUTCH).build()


def preprocess(train_path=ROOT/'data_hf/train.parquet', test_path=ROOT/'data_hf/test.parquet',
               out=ROOT/'full_preprocessed', max_source_rows=None, use_lingua=True):
    """Two-pass, disk-backed pipeline; max_source_rows makes a fast smoke run."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    paths = [Path(train_path), Path(test_path)]
    for path in paths:
        if not path.exists(): raise FileNotFoundError(path)
    meta, users, conflicts = source_metadata(paths)
    users[['user_id','mbti','split']].to_parquet(out/'users.parquet', index=False)
    user_map = users.set_index('author')[['user_id','mbti','split']].to_dict('index')
    overlap = len(set(meta.loc[meta.source_split.eq('train'),'author']) & set(meta.loc[meta.source_split.eq('test'),'author']))
    detector = make_detector() if use_lingua else None
    counts = Counter(); writer = None
    # SQLite tracks exact duplicates without holding all comments in memory.
    db = sqlite3.connect(out/'dedup.sqlite')
    # Rebuild this derived index on each run; otherwise reruns falsely mark all
    # comments as duplicates. The source Parquet files are never modified.
    db.execute('DROP TABLE IF EXISTS hashes')
    db.execute('CREATE TABLE hashes (hash TEXT PRIMARY KEY, first_user TEXT, users INTEGER DEFAULT 1)')
    try:
        for path in paths:
            row_id = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=['author','body','mbti']):
                records = []
                for row in batch.to_pylist():
                    source_id = row_id; row_id += 1
                    if max_source_rows is not None and source_id >= max_source_rows: continue
                    author = row['author']
                    if author in conflicts: counts['conflicting_author_source_rows'] += 1; continue
                    u = user_map.get(author)
                    if u is None: continue
                    segments, suspicious = split_comments(row['body'])
                    if suspicious: counts['suspicious_boundary_rows'] += 1
                    for segment_id, raw in enumerate(segments, 1):
                        text, reason, n_words = clean_comment(raw)
                        high, weak = leakage_flags(text)
                        lang = detect_english(text, detector) if not reason else 'not_checked'
                        if not reason and lang == 'non_english': reason = 'non_english'
                        key = hashlib.sha256(text.casefold().encode()).hexdigest()
                        if not reason:
                            prior = db.execute('SELECT first_user FROM hashes WHERE hash=?',(key,)).fetchone()
                            if prior is None:
                                db.execute('INSERT INTO hashes(hash,first_user) VALUES (?,?)',(key,u['user_id']))
                            elif prior[0] == u['user_id']:
                                reason = 'same_author_duplicate'
                            else:
                                reason = 'cross_author_duplicate'
                                db.execute('UPDATE hashes SET users=2 WHERE hash=?',(key,))
                        counts['raw_comments'] += 1; counts[reason or 'initially_retained'] += 1
                        records.append({'user_id':u['user_id'],'mbti':u['mbti'],'split':u['split'],
                            'source_split':path.stem,'source_row_id':source_id,'segment_id':segment_id,
                            'text':text,'text_hash':key,'word_count':n_words,'language':lang,
                            'leak_high':high,'leak_weak':weak,'reason':reason,'boundary_suspicious':suspicious})
                if records:
                    table = pa.Table.from_pylist(records)
                    if writer is None: writer = pq.ParquetWriter(out/'audit_stage.parquet', table.schema, compression='zstd')
                    writer.write_table(table)
                db.commit()
                if max_source_rows is not None and row_id >= max_source_rows: break
    finally:
        if writer: writer.close()
        db.commit()
    # Second pass removes the first occurrence if the same text later appears
    # under a different author. Three output policies share one audit table.
    writers = {}
    try:
        for batch in pq.ParquetFile(out/'audit_stage.parquet').iter_batches(batch_size=4096):
            frame = batch.to_pandas()
            eligible = frame.reason.eq('')
            keys = frame.loc[eligible,'text_hash'].unique().tolist()
            cross = set()
            for i in range(0,len(keys),900):
                part=keys[i:i+900]
                sql='SELECT hash FROM hashes WHERE users>1 AND hash IN ('+','.join('?' for _ in part)+')'
                cross.update(x[0] for x in db.execute(sql,part))
            frame.loc[eligible & frame.text_hash.isin(cross),'reason']='cross_author_duplicate'
            audit_table = pa.Table.from_pandas(frame, preserve_index=False)
            if 'audit' not in writers:
                writers['audit'] = pq.ParquetWriter(out/'audit.parquet', audit_table.schema, compression='zstd')
            writers['audit'].write_table(audit_table)
            for name, mask in [('classifier_strict',frame.reason.eq('') & ~frame.leak_high),
                               ('classifier_base',frame.reason.eq('')),
                               ('rag_train',frame.reason.eq('') & ~frame.leak_high & frame.split.eq('train'))]:
                part=frame.loc[mask,['user_id','mbti','split','source_row_id','segment_id','text','text_hash','word_count','leak_weak','language']]
                if len(part):
                    tab=pa.Table.from_pandas(part,preserve_index=False)
                    if name not in writers: writers[name]=pq.ParquetWriter(out/f'{name}.parquet',tab.schema,compression='zstd')
                    writers[name].write_table(tab)
                    counts[name]+=len(part)
    finally:
        for w in writers.values():w.close()
        db.close()
    report={'source_rows':len(meta),'source_train_test_author_overlap':overlap,
            'conflicting_authors':len(conflicts),'authors_kept':len(users),
            'max_source_rows_per_file':max_source_rows,'use_lingua':use_lingua,'counts':dict(counts),
            'notes':['body comment boundaries are provisional; inspect boundary_suspicious',
                     'uncertain language is retained, not assumed English',
                     'leak_weak is review-only; high leakage excluded from strict/RAG',
                     'source_row_id and segment_id together identify a source comment']}
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report
