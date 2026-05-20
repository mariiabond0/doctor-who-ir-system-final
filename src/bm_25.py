from rank_bm25 import BM25Okapi
from src.preprocessing import preprocess_text
import json
import sqlite3
import numpy as np
import config

# Cache for BM25 models to avoid rebuilding on each query
_bm25_cache = {}

# def get_cache_key(conn):
#     return str(config.DB_PATH)

"""Return tokenized texts and corresponding doc IDs"""
def build_bm25_corpus(document_corpus):
    texts = []
    doc_ids = []
    for doc_id, doc in document_corpus.items():
        texts.append(preprocess_text(f"{doc['title']} {doc['description']}"))
        doc_ids.append(doc_id)
    return texts, doc_ids

"""Return top_n document IDs ranked by BM25"""
def bm25_search(query: str, texts, doc_ids, top_n=5):
    bm25 = BM25Okapi(texts)
    query_tokens = preprocess_text(query)
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [doc_ids[i] for i in top_indices]

# class BM25Searcher:
#     def __init__(self, texts, doc_ids):
#         self.doc_ids = doc_ids
#         self.bm25 = BM25Okapi(texts)

#     def search(self, query: str, top_n=5):
#         query_tokens = preprocess_text(query)
#         scores = self.bm25.get_scores(query_tokens)

#         top_indices = np.argsort(scores)[::-1][:top_n]
#         return [self.doc_ids[i] for i in top_indices]

def build_bm25_corpus_sqlite(conn):
    """
    Gets preprocessed texts from SQLite and returns texts and doc_ids.
    """
    cur = conn.cursor()
    cur.execute("SELECT doc_id, preprocessed_combined FROM episodes")
    rows = cur.fetchall()
    
    # preprocessed_combined was saved as a JSON array (e.g. ["token1", "token2"]).
    # Parse the JSON to get token lists for BM25.
    texts = [json.loads(row[1]) if row[1] else [] for row in rows]
    doc_ids = [row[0] for row in rows]
    
    return texts, doc_ids


def bm25_search_sqlite(query: str, conn, top_n=5, k1: float = 0.4, b: float = 1.0):
    """
    Performs BM25 search based on a query using the SQLite database.
    Caches the BM25 model for efficiency.
    Optimized parameters: k1=0.4, b=1.0 (from parameter sweep).
    """
    cache_key = (id(conn), float(k1), float(b))
    if cache_key not in _bm25_cache:
        texts, doc_ids = build_bm25_corpus_sqlite(conn)
        # Pass BM25 parameters to the model
        _bm25_cache[cache_key] = (BM25Okapi(texts, k1=k1, b=b), doc_ids)

    bm25, doc_ids = _bm25_cache[cache_key]
    
    query_tokens = preprocess_text(query)
    if not query_tokens:
        return []
    
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [doc_ids[i] for i in top_indices]