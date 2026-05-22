"""Evaluation utilities for Doctor Who IR metrics."""

from typing import Iterable
import math


def compute_metrics(retrieved: Iterable[str], relevant: Iterable[str], top_k: int = 5):
    relevant_set = set(relevant)
    retrieved = list(retrieved)[:top_k]
    # retrieved_set = set(retrieved)
    overlap = sum(1 for doc in retrieved if doc in relevant_set)

    # overlap = len(retrieved_set & relevant_set)
    p_at_k = overlap / len(retrieved) if retrieved else 0.0
    r_at_k = overlap / len(relevant_set) if relevant_set else 0.0

    ap = 0.0
    num_rel = 0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            num_rel += 1
            ap += num_rel / rank
    ap /= len(relevant_set) if relevant_set else 0.0

    mrr = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            mrr = 1.0 / rank
            break

    # nDCG@k with graded relevance based on the ordering in `relevant`.
    # Higher-ranked answers in `relevant` have higher relevance scores.
    graded_relevance = {}
    # assign scores: first (best) -> len(relevant), last -> 1
    for pos, doc_id in enumerate(relevant, start=1):
        graded_relevance[doc_id] = max(len(relevant) - (pos - 1), 0)

    dcg = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        rel = graded_relevance.get(doc_id, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(rank + 1)

    # Ideal DCG: sorted by descending graded relevance
    ideal_rels = sorted(graded_relevance.values(), reverse=True)[:top_k]
    idcg = 0.0
    for rank, rel in enumerate(ideal_rels, start=1):
        idcg += (2**rel - 1) / math.log2(rank + 1)

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "P@5": p_at_k,
        "R@5": r_at_k,
        "AP": ap,
        "MRR": mrr,
        "nDCG": ndcg,
        "overlap": overlap,
    }
