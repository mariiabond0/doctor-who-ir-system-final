import os
import pickle
import sqlite3
from typing import Dict

import config
import numpy as np
from sentence_transformers import SentenceTransformer, util

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

_model = None
_embeddings_cache = {}


def get_model() -> SentenceTransformer:
    """Return a lazily loaded SentenceTransformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(config.MODEL_NAME)
    return _model


def load_embeddings_from_db(conn: sqlite3.Connection) -> Dict[str, np.ndarray]:
    """Load precomputed embeddings from the SQLite database."""
    cur = conn.cursor()
    cur.execute("SELECT doc_id, embedding FROM embeddings")
    rows = cur.fetchall()
    return {row[0]: np.frombuffer(row[1], dtype=np.float32) for row in rows}


def semantic_search_sqlite(query: str, conn: sqlite3.Connection, top_n: int = 5):
    """Vector search over stored embeddings in SQLite."""
    if conn not in _embeddings_cache:
        _embeddings_cache[conn] = load_embeddings_from_db(conn)

    embeddings_dict = _embeddings_cache[conn]
    if not embeddings_dict:
        return []

    doc_ids = list(embeddings_dict.keys())
    corpus_embeddings = np.stack(list(embeddings_dict.values()))

    query_embedding = get_model().encode(query, convert_to_numpy=True, normalize_embeddings=True)
    cosine_scores = util.cos_sim(query_embedding.reshape(1, -1), corpus_embeddings)[0]
    cosine_scores = (
        cosine_scores.cpu().numpy() if hasattr(cosine_scores, "cpu") else np.array(cosine_scores)
    )

    top_indices = np.argsort(cosine_scores)[::-1][:top_n]
    return [doc_ids[i] for i in top_indices]


def encode_corpus(document_corpus):
    """Encode a corpus of documents with a Sentence Transformer."""
    texts = [f"{doc['title']} {doc.get('description', '')}" for doc in document_corpus.values()]
    return get_model().encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def semantic_search(query: str, document_corpus, corpus_embeddings, top_n: int = 5):
    """Semantic search using precomputed embeddings in memory."""
    if corpus_embeddings is None or len(corpus_embeddings) == 0:
        return []

    query_embedding = get_model().encode(query, convert_to_numpy=True, normalize_embeddings=True)
    cosine_scores = util.cos_sim(query_embedding.reshape(1, -1), corpus_embeddings)[0]
    cosine_scores = (
        cosine_scores.cpu().numpy() if hasattr(cosine_scores, "cpu") else np.array(cosine_scores)
    )
    top_indices = np.argsort(cosine_scores)[::-1][:top_n]
    doc_ids = list(document_corpus.keys())
    return [doc_ids[i] for i in top_indices]
