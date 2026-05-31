"""
Two-stage hybrid retrieval and reranking engine for the Doctor Who IR system.
Combines BM25 and Semantic Search using RRF followed by a Cross-Encoder reranker.
"""

import logging
import config
from sentence_transformers import CrossEncoder
from src.bm_25 import bm25_search_sqlite
from src.semantic_search import semantic_search_sqlite

logger = logging.getLogger(__name__)

# Initialize the reranker lazily to save memory during startup
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        logger.info("Loading CrossEncoder reranker model...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def fused_query(query, conn, top_k=config.DEFAULT_TOP_K, k=10, candidate_k=50, rerank_k=20):
    """
    Executes a two-stage hybrid retrieval pipeline.
    1. Fetches candidate lists via lexical (BM25) and dense semantic methods.
    2. Blends results using Reciprocal Rank Fusion (RRF).
    3. Reranks the top candidates using a high-accuracy neural Cross-Encoder.
    """
    # Get results from BM25 (sparse) and semantic (dense)
    bm25_results = bm25_search_sqlite(query, conn, top_n=candidate_k)
    semantic_results = semantic_search_sqlite(query, conn, top_n=candidate_k)

    # Create rank dictionaries
    bm25_ranks = {doc: rank + 1 for rank, doc in enumerate(bm25_results)}
    semantic_ranks = {doc: rank + 1 for rank, doc in enumerate(semantic_results)}

    # Combine using Reciprocal Rank Fusion (RRF)
    fused_scores = {}
    all_docs = set(bm25_results) | set(semantic_results)
    for doc in all_docs:
        score = 0.0
        if doc in bm25_ranks:
            score += 1 / (k + bm25_ranks[doc])
        if doc in semantic_ranks:
            score += 1 / (k + semantic_ranks[doc])
        fused_scores[doc] = score

    # Sort by fused score descending
    sorted_docs = sorted(fused_scores, key=fused_scores.get, reverse=True)

    # Gather top candidates for deeper neural reranking
    top_candidates = sorted_docs[:rerank_k]
    reranker = get_reranker()

    # Fetch text context for the cross-encoder pair matching
    cur = conn.cursor()
    doc_texts = []
    for doc_id in top_candidates:
        cur.execute("SELECT title, description, summary FROM episodes WHERE doc_id=?", (doc_id,))
        row = cur.fetchone()
        if row:
            text = " ".join([str(x) for x in row if x])
        else:
            text = doc_id
        doc_texts.append(text)

    # Score absolute query-document text pairs
    pairs = [(query, text) for text in doc_texts]
    scores = reranker.predict(pairs)

    # Re-sort using high-fidelity reranker scores
    reranked = sorted(zip(top_candidates, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in reranked[:top_k]]
