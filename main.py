"""Doctor Who Information Retrieval System - Main evaluation script."""

import argparse
import json
import logging
import sqlite3
import pandas as pd

import faiss
from sentence_transformers import SentenceTransformer

import config
from src.bm_25 import bm25_search_sqlite
from src.boolean_search import boolean_search_sqlite
from src.evaluation import compute_metrics
from src.semantic_search import semantic_search_sqlite

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)

with open(config.QUERIES_PATH, "r", encoding="utf-8") as f:
    second_example_20_queries = json.load(f)

QUERIES = second_example_20_queries["queries"]
ANSWERS = second_example_20_queries["answers"]


def load_faiss_mapping(mapping_path):
    with open(mapping_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return {int(k): v for k, v in data.items()}


def faiss_query(query, index, mapping, model, top_k=config.DEFAULT_TOP_K):
    query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype(
        "float32"
    )
    distances, indices = index.search(query_emb.reshape(1, -1), top_k)
    return [mapping[i] for i in indices[0] if int(i) in mapping]


def fused_query(query, conn, top_k=config.DEFAULT_TOP_K, k=60, candidate_k=50):
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
        rrf_sparse = 1 / (k + bm25_ranks.get(doc, candidate_k + 1))
        rrf_dense = 1 / (k + semantic_ranks.get(doc, candidate_k + 1))
        fused_scores[doc] = rrf_sparse + rrf_dense

    # Sort by fused score descending
    sorted_docs = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return sorted_docs[:top_k]


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


def save_results(summary, out_path=None):
    import pandas as pd

    if out_path is None:
        out_path = config.RESULTS_CSV_PATH
    df = pd.DataFrame(summary)
    df.to_csv(out_path, index=False)
    logger.info("Saved results to %s", out_path)


def main():
    conn = sqlite3.connect(str(config.DB_PATH))
    logger.info(f"Connected to database: {config.DB_PATH}")

    try:
        faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        faiss_index.hnsw.efSearch = config.FAISS_EF_SEARCH
        faiss_mapping = load_faiss_mapping(config.FAISS_MAPPING_PATH)
        faiss_model = SentenceTransformer(config.MODEL_NAME)
    except Exception as error:
        logger.error("Failed to load FAISS resources: %s", error)
        raise

    methods = [
        ("Boolean Search", lambda q: boolean_search_sqlite(q, conn, top_n=config.DEFAULT_TOP_K)),
        ("BM25 Search", lambda q: bm25_search_sqlite(q, conn, top_n=config.DEFAULT_TOP_K)),
        ("Semantic Search", lambda q: semantic_search_sqlite(q, conn, top_n=config.DEFAULT_TOP_K)),
        (
            "FAISS Semantic Search",
            lambda q: faiss_query(
                q, faiss_index, faiss_mapping, faiss_model, top_k=config.DEFAULT_TOP_K
            ),
        ),
        ("Fused Search", lambda q: fused_query(q, conn, top_k=config.DEFAULT_TOP_K)),
    ]

    results = []
    for name, fn in methods:
        method_metrics = evaluate_method(name, fn)
        for i, metrics in enumerate(method_metrics):
            row = {
                # "query": QUERIES[i],
                # include query number in the method name
                "method": f"{name} {i+1}",
                # "expected": ";".join(ANSWERS[i]),
                "overlap": metrics["overlap"],
                "P@5": metrics["P@5"],
                "R@5": metrics["R@5"],
                "AP": metrics["AP"],
                "MRR": metrics["MRR"],
                "nDCG": metrics.get("nDCG", 0.0),
            }
            results.append(row)

    save_results(results)
    logger.info("Evaluation complete.")


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


parser = argparse.ArgumentParser(description="Doctor Who IR evaluation system")

parser.add_argument(
    "--sweep-bm25",
    action="store_true",
    help="Run BM25 parameter sweep",
)

parser.add_argument(
    "--k1-values",
    type=float,
    nargs="+",
    default=[0.5, 1.0, 1.5, 2.0],
)

parser.add_argument(
    "--b-values",
    type=float,
    nargs="+",
    default=[0.4, 0.6, 0.75, 1.0],
)

parser.add_argument(
    "--output",
    type=str,
    default="bm25_testing.csv",
)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.sweep_bm25:
        conn = sqlite3.connect(str(config.DB_PATH))
        logger.info(f"Connected to database: {config.DB_PATH}")
        out_file = config.DATA_DIR / args.output
        logger.info(f"Running BM25 parameter sweep with k1={args.k1_values}, b={args.b_values}")
        bm25_param_sweep(conn, args.k1_values, args.b_values, out_path=out_file)
        logger.info(f"Sweep complete. Results saved to %s", out_file)
        conn.close()
    else:
        main()
