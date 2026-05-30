from rank_bm25 import BM25Okapi
from src.preprocessing import preprocess_text
import sqlite3
import numpy as np
import json

# Cache for BM25 models to avoid rebuilding on each query
_bm25_cache = {}


def build_bm25_corpus(document_corpus):
    """
    Build BM25 corpus from in-memory document dictionary.
    """
    texts = []
    doc_ids = []
    for doc_id, doc in document_corpus.items():
        texts.append(preprocess_text(f"{doc['title']} {doc['description']} {doc['summary']} "))
        doc_ids.append(doc_id)
    return texts, doc_ids


def bm25_search(query: str, texts, doc_ids, top_n=5):
    """
    Run BM25 ranking on prebuilt corpus.
    """
    bm25 = BM25Okapi(texts)
    query_tokens = preprocess_text(query)
    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:top_n]
    return [doc_ids[i] for i in top_indices]


def build_bm25_corpus_sqlite(conn):
    cur = conn.cursor()
    cur.execute("SELECT doc_id, preprocessed_combined FROM episodes")
    rows = cur.fetchall()

    texts = []
    doc_ids = []

    for doc_id, blob in rows:
        if not blob:
            texts.append([])
        else:
            texts.append(json.loads(blob))  # JSON-encoded token list from the DB

        doc_ids.append(doc_id)
    return texts, doc_ids


def bm25_search_sqlite(query: str, conn, top_n=5, k1: float = 1.25, b: float = 0.6):
    """
    BM25 search over SQLite-backed corpus with caching.
    """
    cache_key = (id(conn), float(k1), float(b))

    if cache_key not in _bm25_cache:
        texts, doc_ids = build_bm25_corpus_sqlite(conn)
        bm25_model = BM25Okapi(texts, k1=k1, b=b)
        _bm25_cache[cache_key] = (bm25_model, doc_ids)

    bm25, doc_ids = _bm25_cache[cache_key]

    query_tokens = preprocess_text(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_n]

    return [doc_ids[i] for i in top_indices]
