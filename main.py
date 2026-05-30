"""Doctor Who Information Retrieval System - Main evaluation script."""

import argparse
import logging
import sqlite3
import pandas as pd

import faiss
from sentence_transformers import SentenceTransformer

import config
from src.bm_25 import bm25_search_sqlite
from src.boolean_search import boolean_search_sqlite
from src.evaluation import evaluate_method, bm25_param_sweep, QUERIES, ANSWERS
from src.semantic_search import semantic_search_sqlite
from src.faiss_search import faiss_query, load_faiss_mapping
from src.fused_search import fused_query
from src.rag import rag_query, format_rag_output

logger = logging.getLogger(__name__)


def save_results(summary, out_path=None):
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
                "method": name,
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


parser = argparse.ArgumentParser(description="Doctor Who IR evaluation system")

parser.add_argument(
    "--mode",
    type=str,
    choices=["eval", "rag", "search"],
    default="eval",
    help="Operation mode: 'eval' for evaluation, 'rag' for RAG answering, 'search' for search demo"
)

parser.add_argument(
    "--query",
    type=str,
    default=None,
    help="Query string for RAG or search mode"
)

parser.add_argument(
    "--search-method",
    type=str,
    choices=["boolean", "bm25", "semantic", "faiss", "fused"],
    default="boolean",
    help="Search method to use in search mode (default: boolean)"
)

parser.add_argument(
    "--sweep-bm25",
    action="store_true",
    help="Run BM25 parameter sweep",
)

parser.add_argument(
    "--k1-values",
    type=float,
    nargs="+",
    default=[0.5, 1.0, 1.25, 1.5, 1.75, 2.0],
)

parser.add_argument(
    "--b-values",
    type=float,
    nargs="+",
    default=[0.4, 0.6, 0.75, 0.85, 0.9, 1.0],
)

parser.add_argument(
    "--output",
    type=str,
    default="bm25_testing.csv",
)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.mode == "rag":
        if not args.query:
            print("Error: --query required for RAG mode")
            print("Usage: python main.py --mode rag --query 'your question'")
            exit(1)
        
        logger.info(f"RAG Mode: {args.query}")
        result = rag_query(args.query)
        print(format_rag_output(result))
        
    elif args.mode == "search":
        if not args.query:
            print("Error: --query required for search mode")
            print("Usage: python main.py --mode search --query 'your query' --search-method [boolean|semantic|bm25|faiss|fused]")
            exit(1)
        
        conn = sqlite3.connect(str(config.DB_PATH))
        logger.info(f"Search Mode: {args.query} using {args.search_method}")
        
        search_methods = {
            "boolean": lambda q: boolean_search_sqlite(q, conn, top_n=config.DEFAULT_TOP_K),
            "bm25": lambda q: bm25_search_sqlite(q, conn, top_n=config.DEFAULT_TOP_K),
            "semantic": lambda q: semantic_search_sqlite(q, conn, top_n=config.DEFAULT_TOP_K),
        }
        
        if args.search_method == "faiss":
            faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))
            faiss_index.hnsw.efSearch = config.FAISS_EF_SEARCH
            faiss_mapping = load_faiss_mapping(config.FAISS_MAPPING_PATH)
            faiss_model = SentenceTransformer(config.MODEL_NAME)
            search_fn = lambda q: faiss_query(q, faiss_index, faiss_mapping, faiss_model, top_k=config.DEFAULT_TOP_K)
        elif args.search_method == "fused":
            search_fn = lambda q: fused_query(q, conn, top_k=config.DEFAULT_TOP_K)
        else:
            search_fn = search_methods[args.search_method]
        
        results = search_fn(args.query)
        print(f"\n{'='*80}")
        print(f"Search Results for: {args.query}")
        print(f"Method: {args.search_method.upper()}")
        print(f"{'='*80}\n")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc}\n")
        conn.close()
        
    elif args.sweep_bm25:
        conn = sqlite3.connect(str(config.DB_PATH))
        logger.info(f"Connected to database: {config.DB_PATH}")
        out_file = config.DATA_DIR / args.output
        logger.info(f"Running BM25 parameter sweep with k1={args.k1_values}, b={args.b_values}")
        bm25_param_sweep(conn, args.k1_values, args.b_values, out_path=out_file)
        logger.info(f"Sweep complete. Results saved to %s", out_file)
        conn.close()
    else:
        main()
