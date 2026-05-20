"""Doctor Who Information Retrieval System - Main evaluation script."""

import argparse
import json
import logging
import sqlite3

import faiss
from sentence_transformers import SentenceTransformer

import config
from src.bm_25 import bm25_search_sqlite
from src.boolean_search import boolean_search_sqlite
from src.creating_corpus import filter_seasons, load_episode_data
from src.evaluation import compute_metrics
from src.knowledge_graph import (
    KnowledgeGraph,
    document_to_graph,
    kg_search,
    load_gliner2_model,
    merge_graphs,
    summarize_kg_results,
)
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
    query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
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
    print(f"Mean P@5: {mean_p5:.2f}, Mean R@5: {mean_r5:.2f}, MAP: {mean_ap:.2f}, MRR: {mean_mrr:.2f}, Mean nDCG: {mean_ndcg:.3f}")
    return metrics


def save_results(summary, out_path=None):
    import pandas as pd

    if out_path is None:
        out_path = config.RESULTS_CSV_PATH
    df = pd.DataFrame(summary)
    df.to_csv(out_path, index=False)
    logger.info("Saved results to %s", out_path)


def build_knowledge_graph(model_name: str, entity_types=None, relation_types=None, threshold: float = 0.5):
    logger.info("Loading episode data for knowledge graph construction")
    df = load_episode_data()
    df = filter_seasons(df)

    graphs = []
    kg_model = load_gliner2_model(model_name)

    for row in df.itertuples(index=False):
        title = str(getattr(row, "title", "")).strip()
        description = str(getattr(row, "description", "")).strip()
        text = " ".join(filter(None, [title, description])).strip()
        source_id = f"{int(getattr(row, 'season', 0))}x{int(getattr(row, 'number', 0))}"
        graph = document_to_graph(
            text,
            kg_model,
            entity_types=entity_types,
            relation_types=relation_types,
            threshold=threshold,
            source_id=source_id,
        )
        graphs.append(graph)

    merged_graph = merge_graphs(graphs)
    logger.info("Knowledge graph built from %d documents", len(graphs))
    return merged_graph


def save_knowledge_graph(graph: KnowledgeGraph, output_path: str):
    graph.save_json(output_path)
    logger.info("Saved knowledge graph to %s", output_path)


def load_knowledge_graph(path: str) -> KnowledgeGraph:
    return KnowledgeGraph.load_json(path)


def query_knowledge_graph(path: str, subject: str = None, relation: str = None, object_: str = None):
    graph = load_knowledge_graph(path)
    matches = graph.query(subject=subject, relation=relation, object=object_)
    logger.info("Found %d matching edges", len(matches))
    return matches


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

    kg_graph = None
    kg_model = None
    kg_summary_results = []
    kg_path = config.DATA_DIR / "knowledge_graph.json"
    if kg_path.exists():
        try:
            kg_graph = KnowledgeGraph.load_json(str(kg_path))
            kg_model = load_gliner2_model("gliner2/gliner2-base")
            logger.info("Loaded knowledge graph from %s", kg_path)
        except Exception as error:
            logger.warning("Unable to load KG graph or model: %s", error)
            kg_graph = None
            kg_model = None

    def kg_query_fn(query):
        docs, edges = kg_search(query, kg_graph, kg_model, top_k=config.DEFAULT_TOP_K)
        kg_summary_results.append(summarize_kg_results(query, docs, edges))
        return docs

    methods = [
        ("Boolean Search", lambda q: boolean_search_sqlite(q, conn, top_n=config.DEFAULT_TOP_K)),
        ("BM25 Search", lambda q: bm25_search_sqlite(q, conn, top_n=config.DEFAULT_TOP_K)),
        ("Semantic Search", lambda q: semantic_search_sqlite(q, conn, top_n=config.DEFAULT_TOP_K)),
        ("FAISS Semantic Search", lambda q: faiss_query(q, faiss_index, faiss_mapping, faiss_model, top_k=config.DEFAULT_TOP_K)),
        ("Fused Search", lambda q: fused_query(q, conn, top_k=config.DEFAULT_TOP_K)),
    ]
    if kg_graph is not None and kg_model is not None:
        methods.append(("Knowledge Graph Search", kg_query_fn))

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
    if kg_summary_results:
        import json

        kg_summary_path = config.DATA_DIR / "kg_search_summary.json"
        with open(kg_summary_path, "w", encoding="utf-8") as outfile:
            json.dump(kg_summary_results, outfile, ensure_ascii=False, indent=2)
        logger.info("Saved KG search summary to %s", kg_summary_path)
    logger.info("Evaluation complete.")


def kg_only_search(kg_graph_path: str, kg_model_name: str, kg_results_path: str, entity_types=None, relation_types=None, threshold: float = 0.5):
    kg_graph = KnowledgeGraph.load_json(kg_graph_path)
    kg_model = load_gliner2_model(kg_model_name)

    summary_results = []
    results = []

    def kg_query_fn(query):
        docs, edges = kg_search(query, kg_graph, kg_model, entity_types=entity_types, relation_types=relation_types, threshold=threshold, top_k=config.DEFAULT_TOP_K)
        summary_results.append(summarize_kg_results(query, docs, edges))
        return docs

    method_metrics = evaluate_method("Knowledge Graph Search", kg_query_fn)
    for i, metrics in enumerate(method_metrics):
        row = {
            "method": f"Knowledge Graph Search {i+1}",
            "overlap": metrics["overlap"],
            "P@5": metrics["P@5"],
            "R@5": metrics["R@5"],
            "AP": metrics["AP"],
            "MRR": metrics["MRR"],
            "nDCG": metrics.get("nDCG", 0.0),
        }
        results.append(row)

    save_results(results, out_path=kg_results_path)
    import json

    kg_summary_path = config.DATA_DIR / "kg_search_summary.json"
    with open(kg_summary_path, "w", encoding="utf-8") as outfile:
        json.dump(summary_results, outfile, ensure_ascii=False, indent=2)
    logger.info("Saved KG search summary to %s", kg_summary_path)
    logger.info("KG-only search complete.")


def bm25_param_sweep(conn, k1_values, b_values, top_k=config.DEFAULT_TOP_K, out_path=None):
    """Run grid search over BM25 k1 and b values and save aggregated metrics.

    Writes a CSV with one row per (k1,b) containing mean P@5, mean R@5 (MAR), MAP, MRR, mean nDCG.
    """
    import pandas as pd

    sweep_rows = []
    for k1 in k1_values:
        for b in b_values:
            name = f"BM25 k1={k1} b={b}"
            print(f"Running sweep: {name}")
            method_metrics = evaluate_method(name, lambda q: bm25_search_sqlite(q, conn, top_n=top_k, k1=k1, b=b))

            mean_p5 = sum(m["P@5"] for m in method_metrics) / len(method_metrics)
            mean_r5 = sum(m["R@5"] for m in method_metrics) / len(method_metrics)
            mean_ap = sum(m["AP"] for m in method_metrics) / len(method_metrics)
            mean_mrr = sum(m["MRR"] for m in method_metrics) / len(method_metrics)
            mean_ndcg = sum(m.get("nDCG", 0.0) for m in method_metrics) / len(method_metrics)

            sweep_rows.append({
                "method": name,
                "k1": k1,
                "b": b,
                "mean_P@5": mean_p5,
                "mean_R@5": mean_r5,
                "MAR": mean_r5,
                "MAP": mean_ap,
                "mean_MRR": mean_mrr,
                "mean_nDCG": mean_ndcg,
            })

    df = pd.DataFrame(sweep_rows)
    if out_path is None:
        out_path = config.RESULTS_CSV_PATH.with_name("bm25_param_tuning.csv")
    df.to_csv(out_path, index=False)
    logger.info("Saved BM25 sweep results to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Doctor Who IR System - Evaluation, KG build, and tuning")
    parser.add_argument("--sweep-bm25", action="store_true", help="Run BM25 parameter sweep")
    parser.add_argument("--k1-values", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0],
                        help="List of k1 values to sweep (default: 0.5 1.0 1.5 2.0)")
    parser.add_argument("--b-values", type=float, nargs="+", default=[0.4, 0.6, 0.75, 1.0],
                        help="List of b values to sweep (default: 0.4 0.6 0.75 1.0)")
    parser.add_argument("--output", type=str, default="bm25_testing.csv",
                        help="Output CSV filename for sweep results (default: bm25_testing.csv)")
    parser.add_argument("--build-kg", action="store_true", help="Build a knowledge graph from the episode corpus")
    parser.add_argument("--kg-model", type=str, default="gliner2/gliner2-base",
                        help="GliNER2 model repo or local path to load for KG extraction")
    parser.add_argument("--kg-output", type=str, default="knowledge_graph.json",
                        help="Output filename for built knowledge graph")
    parser.add_argument("--kg-search", action="store_true", help="Run KG-only search over queries and save results")
    parser.add_argument("--kg-results-output", type=str, default="kg_search_results_summary.csv",
                        help="Output filename for KG-only search results")
    parser.add_argument("--kg-threshold", type=float, default=0.5,
                        help="Confidence threshold for entity/relation extraction")
    parser.add_argument("--kg-query-subject", type=str, default=None,
                        help="Subject text to query knowledge graph edges")
    parser.add_argument("--kg-query-relation", type=str, default=None,
                        help="Relation type to query knowledge graph edges")
    parser.add_argument("--kg-query-object", type=str, default=None,
                        help="Object text to query knowledge graph edges")
    parser.add_argument("--entity-types", type=str, nargs="*", default=None,
                        help="Entity types to extract from the corpus")
    parser.add_argument("--relation-types", type=str, nargs="*", default=None,
                        help="Relation types to extract from the corpus")

    args = parser.parse_args()

    if args.sweep_bm25:
        conn = sqlite3.connect(str(config.DB_PATH))
        logger.info(f"Connected to database: {config.DB_PATH}")
        out_file = config.DATA_DIR / args.output
        logger.info(f"Running BM25 parameter sweep with k1={args.k1_values}, b={args.b_values}")
        bm25_param_sweep(conn, args.k1_values, args.b_values, out_path=out_file)
        logger.info(f"Sweep complete. Results saved to %s", out_file)
        conn.close()
    elif args.build_kg:
        out_file = config.DATA_DIR / args.kg_output
        kg = build_knowledge_graph(
            model_name=args.kg_model,
            entity_types=args.entity_types,
            relation_types=args.relation_types,
            threshold=args.kg_threshold,
        )
        save_knowledge_graph(kg, str(out_file))
        logger.info("Knowledge graph build complete.")
    elif args.kg_search:
        kg_graph_path = config.DATA_DIR / args.kg_output
        if not kg_graph_path.exists():
            logger.error("Knowledge graph file not found: %s", kg_graph_path)
            raise FileNotFoundError(kg_graph_path)
        out_file = config.DATA_DIR / args.kg_results_output
        kg_only_search(
            str(kg_graph_path),
            args.kg_model,
            str(out_file),
            entity_types=args.entity_types,
            relation_types=args.relation_types,
            threshold=args.kg_threshold,
        )
    elif args.kg_query_subject or args.kg_query_relation or args.kg_query_object:
        out_file = config.DATA_DIR / args.kg_output
        matches = query_knowledge_graph(
            str(out_file),
            subject=args.kg_query_subject,
            relation=args.kg_query_relation,
            object_=args.kg_query_object,
        )
        for edge in matches:
            print(edge)
        logger.info("Knowledge graph query complete. %d matches found.", len(matches))
    else:
        main()
