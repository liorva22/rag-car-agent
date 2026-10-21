"""Postgres index: full-text search (tsvector) + pgvector dense search over one table, with tag/source filters."""

import os
import time
from functools import cache

import numpy as np

from .corpus import ROOT, load_corpus, tags

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
DIM = 384

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS docs (
    id text PRIMARY KEY,
    source text NOT NULL,
    title text NOT NULL,
    component text,
    date text,
    body text NOT NULL,
    url text,
    vehicles text[] NOT NULL,
    tags text[] NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || body)) STORED,
    embedding vector({DIM}) NOT NULL
);
CREATE INDEX IF NOT EXISTS docs_tsv ON docs USING gin (tsv);
CREATE INDEX IF NOT EXISTS docs_tags ON docs USING gin (tags);
"""
# ponytail: exact vector scan, no HNSW. 9k rows scan in a few ms and filtered HNSW can return fewer than k rows.
# Add an HNSW index with hnsw.iterative_scan once the corpus is past ~100k rows.

COLS = "id, source, title, component, date, body, url, vehicles"


EMBED_CHARS = 1000  # ~250 tokens: title + start of the record carries the topic; the full text stays in the keyword index
EMB_CACHE = ROOT / "data" / "embeddings.npz"


def doc_text(d: dict) -> str:
    return f"{d['title']}\n{d['text']}"


def embed_docs(docs: list[dict], batch: int = 32) -> np.ndarray:
    """Passage embeddings, reusing data/embeddings.npz (committed) for docs it already covers."""
    cached = {}
    if EMB_CACHE.exists():
        z = np.load(EMB_CACHE)
        if str(z["model"]) == EMBED_MODEL:
            cached = dict(zip(z["ids"].tolist(), z["vecs"], strict=True))
    todo = [d for d in docs if d["id"] not in cached]
    if todo:
        t0 = time.perf_counter()
        vecs = embedder().passage_embed([doc_text(d)[:EMBED_CHARS] for d in todo], batch_size=batch)
        cached |= {d["id"]: v.astype(np.float16) for d, v in zip(todo, vecs, strict=True)}
        print(f"embedded {len(todo)} docs in {time.perf_counter() - t0:.0f}s")
    return np.stack([cached[d["id"]] for d in docs]).astype(np.float32)


def save_embeddings(docs: list[dict] | None = None) -> None:
    docs = docs if docs is not None else load_corpus()
    vecs = embed_docs(docs).astype(np.float16)
    np.savez_compressed(EMB_CACHE, ids=np.array([d["id"] for d in docs]), vecs=vecs, model=np.array(EMBED_MODEL))


@cache
def embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(EMBED_MODEL)


@cache
def reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(RERANK_MODEL)


def embed_query(q: str) -> np.ndarray:
    return next(iter(embedder().query_embed(q)))


def connect():
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def count(conn) -> int:
    conn.execute(SCHEMA)
    return conn.execute("SELECT count(*) FROM docs").fetchone()[0]


def load(conn, docs: list[dict] | None = None, batch: int = 256) -> int:
    """(Re)build the index from the corpus. Embeddings come from data/embeddings.npz; missing ones are computed locally."""
    docs = docs if docs is not None else load_corpus()
    conn.execute(SCHEMA)
    conn.execute("TRUNCATE docs")
    t0 = time.perf_counter()
    all_vecs = embed_docs(docs)
    for i in range(0, len(docs), batch):
        chunk, vecs = docs[i : i + batch], all_vecs[i : i + batch]
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO docs ({COLS}, tags, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    (d["id"], d["source"], d["title"], d["component"], d["date"], d["text"], d["url"], d["vehicles"], tags(d), v)
                    for d, v in zip(chunk, vecs, strict=True)
                ],
            )
    print(f"indexed {len(docs)} docs in {time.perf_counter() - t0:.0f}s")
    return len(docs)


def _where(tag: str | None, source: str | None) -> tuple[str, list]:
    sql, args = [], []
    if tag:
        sql.append("%s = ANY(tags)")
        args.append(tag)
    if source:
        sql.append("source = %s")
        args.append(source)
    return (" AND " + " AND ".join(sql)) if sql else "", args


def keyword_search(conn, q: str, k: int, tag=None, source=None) -> list[tuple[str, float]]:
    # OR the query terms: plainto_tsquery ANDs them, which returns nothing for most natural questions.
    w, args = _where(tag, source)
    return conn.execute(
        "WITH q AS (SELECT replace(plainto_tsquery('english', %s)::text, '&', '|') AS t) "
        f"SELECT id, ts_rank_cd(tsv, t::tsquery, 1) AS s FROM docs, q WHERE t <> '' AND tsv @@ t::tsquery{w} "
        "ORDER BY s DESC, id LIMIT %s",
        [q, *args, k],
    ).fetchall()


def dense_search(conn, qvec: np.ndarray, k: int, tag=None, source=None) -> list[tuple[str, float]]:
    w, args = _where(tag, source)
    return conn.execute(
        f"SELECT id, 1 - (embedding <=> %s) AS s FROM docs WHERE true{w} ORDER BY embedding <=> %s, id LIMIT %s",
        [qvec, *args, qvec, k],
    ).fetchall()


def fetch(conn, ids: list[str]) -> dict[str, dict]:
    rows = conn.execute(f"SELECT {COLS} FROM docs WHERE id = ANY(%s)", [ids]).fetchall()
    keys = ["id", "source", "title", "component", "date", "text", "url", "vehicles"]
    return {r[0]: dict(zip(keys, r, strict=True)) for r in rows}
