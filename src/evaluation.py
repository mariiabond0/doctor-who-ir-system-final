"""Evaluation utilities for Doctor Who IR metrics."""

import math
import json
import config
import logging

from typing import Iterable
from src.bm_25 import bm25_search_sqlite

logger = logging.getLogger(__name__)

with open(config.QUERIES_PATH, "r", encoding="utf-8") as f:
    queries_data = json.load(f)

QUERIES = queries_data["queries"]
ANSWERS = queries_data["answers"]


def compute_metrics(retrieved: Iterable[str], relevant: Iterable[str], top_k: int = 5):
    relevant_set = set(relevant)
    retrieved = list(retrieved)[:top_k]
    overlap = sum(1 for doc in retrieved if doc in relevant_set)

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


def evaluate_method(name, query_fn):
    print(f"\n--- {name} ---")
    metrics = []
    for i, query in enumerate(QUERIES):
        retrieved = query_fn(query)
        result = compute_metrics(retrieved, ANSWERS[i], top_k=config.DEFAULT_TOP_K)
        metrics.append(result)
        print(
            f"Query {i+1}: Overlap {result['overlap']}/{len(ANSWERS[i])}, "
            f"P@5 {result['P@5']:.2f}, R@5 {result['R@5']:.2f}, "
            f"AP {result['AP']:.2f}, MRR {result['MRR']:.2f}, nDCG {result.get('nDCG',0.0):.3f}"
        )

    mean_p5 = sum(m["P@5"] for m in metrics) / len(metrics)
    mean_r5 = sum(m["R@5"] for m in metrics) / len(metrics)
    mean_ap = sum(m["AP"] for m in metrics) / len(metrics)
    mean_mrr = sum(m["MRR"] for m in metrics) / len(metrics)
    mean_ndcg = sum(m.get("nDCG", 0.0) for m in metrics) / len(metrics)
    print(
        f"Mean P@5: {mean_p5:.2f}, Mean R@5: {mean_r5:.2f}, MAP: {mean_ap:.2f}, MRR: {mean_mrr:.2f}, Mean nDCG: {mean_ndcg:.3f}"
    )
    return metrics


def bm25_param_sweep(conn, k1_values, b_values, top_k=config.DEFAULT_TOP_K, out_path=None):
    """Run grid search over BM25 k1 and b values and save aggregated metrics.

    Writes a CSV with one row per (k1,b) containing mean P@5, mean R@5 (MAR), MAP, MRR, mean nDCG.
    """
    import pandas as pd

    sweep_rows = []
    for k1 in k1_values:
        for b in b_values:
            name = f"BM25 k1={k1} b={b}"
            method_metrics = evaluate_method(
                name, lambda q: bm25_search_sqlite(q, conn, top_n=top_k, k1=k1, b=b)
            )

            mean_p5 = sum(m["P@5"] for m in method_metrics) / len(method_metrics)
            mean_r5 = sum(m["R@5"] for m in method_metrics) / len(method_metrics)
            mean_ap = sum(m["AP"] for m in method_metrics) / len(method_metrics)
            mean_mrr = sum(m["MRR"] for m in method_metrics) / len(method_metrics)
            mean_ndcg = sum(m.get("nDCG", 0.0) for m in method_metrics) / len(method_metrics)

            sweep_rows.append(
                {
                    "method": name,
                    "k1": k1,
                    "b": b,
                    "mean_P@5": mean_p5,
                    "mean_R@5": mean_r5,
                    "MAR": mean_r5,
                    "MAP": mean_ap,
                    "mean_MRR": mean_mrr,
                    "mean_nDCG": mean_ndcg,
                }
            )

    df = pd.DataFrame(sweep_rows)
    if out_path is None:
        out_path = config.RESULTS_CSV_PATH.with_name("bm25_param_tuning.csv")
    df.to_csv(out_path, index=False)
    logger.info("Saved BM25 sweep results to %s", out_path)


