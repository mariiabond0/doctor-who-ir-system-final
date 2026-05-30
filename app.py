"""
Production-safe Flask IR application (refactored)
Improvements:
- Context-managed SQLite access
- Safer FAISS abstraction layer
- Unified search router
- Cleaner API contracts
- Safer exception handling
"""

import sqlite3
import logging
import json
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Tuple

from flask import Flask, render_template, request, jsonify, send_from_directory
import faiss
from sentence_transformers import SentenceTransformer

import config
from src.bm_25 import bm25_search_sqlite
from src.boolean_search import boolean_search_sqlite
from src.semantic_search import semantic_search_sqlite
from src.rag import rag_query
from src.fused_search import fused_query

# -------------------------
# App setup
# -------------------------

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['JSON_SORT_KEYS'] = False

logger = logging.getLogger(__name__)

# -------------------------
# Context-managed DB
# -------------------------

@contextmanager
def get_db():
    conn = sqlite3.connect(str(config.DB_PATH))
    try:
        yield conn
    finally:
        conn.close()

# -------------------------
# FAISS abstraction layer
# -------------------------

class FaissService:
    def __init__(self):
        self._index = None
        self._mapping = None
        self._model = None

    def load(self):
        if self._index is not None:
            return

        self._index = faiss.read_index(str(config.FAISS_INDEX_PATH))

        if hasattr(self._index, "hnsw"):
            self._index.hnsw.efSearch = config.FAISS_EF_SEARCH

        with open(config.FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
            self._mapping = {int(k): v for k, v in raw.items()}

        self._model = SentenceTransformer(config.MODEL_NAME)

        logger.info("FAISS service initialized")

    def search(self, query: str, top_k: int) -> List[Any]:
        self.load()

        try:
            emb = self._model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype("float32")

            distances, indices = self._index.search(emb.reshape(1, -1), top_k)

            results = []
            for i in indices[0]:
                if i >= 0 and i in self._mapping:
                    results.append(self._mapping[i])

            return results

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []


faiss_service = FaissService()

# -------------------------
# Search router
# -------------------------

SEARCH_METHODS = {"boolean", "bm25", "semantic", "faiss", "fused", "rag"}


def run_search(method: str, query: str, conn, top_k: int):
    if method == "boolean":
        return boolean_search_sqlite(query, conn, top_n=top_k)

    if method == "bm25":
        return bm25_search_sqlite(query, conn, top_n=top_k)

    if method == "semantic":
        return semantic_search_sqlite(query, conn, top_n=top_k)

    if method == "faiss":
        return faiss_service.search(query, top_k)

    if method == "fused":
        return fused_query(query, conn, top_k=top_k)

    raise ValueError(f"Unknown method: {method}")

# -------------------------
# API
# -------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/methods')
def methods():
    return jsonify([
        {"id": "boolean", "name": "Boolean", "speed": "Very Fast"},
        {"id": "bm25", "name": "BM25", "speed": "Fast"},
        {"id": "semantic", "name": "Semantic", "speed": "Medium"},
        {"id": "faiss", "name": "FAISS", "speed": "Very Fast"},
        {"id": "fused", "name": "Fused", "speed": "Medium"},
        {"id": "rag", "name": "RAG", "speed": "Slow"},
    ])


@app.route('/api/search', methods=['POST'])
def search():
    data = request.get_json(silent=True) or {}

    query = (data.get("query") or "").strip()
    method = data.get("method", "boolean")

    if not query:
        return jsonify({"error": "query required"}), 400

    if len(query) < 2:
        return jsonify({"error": "query too short"}), 400

    if method not in SEARCH_METHODS:
        return jsonify({"error": "invalid method"}), 400
    
    logger.info(f"Search request: method={method}, query='{query}'")

    if method == "rag":
        return jsonify({"error": "use /api/rag"}), 400

    try:
        with get_db() as conn:
            # 1. Get raw doc_ids from your search algorithms
            results = run_search(method, query, conn, config.DEFAULT_TOP_K)

            # 2. Enrich doc_ids with title and description from the 'episodes' table
            enriched_results = []
            cursor = conn.cursor()
            
            for item in results:
                # Some search methods might return dictionary elements or full objects; 
                # extract the doc_id safely regardless of format.
                doc_id = item["doc_id"] if isinstance(item, dict) and "doc_id" in item else str(item)
                
                cursor.execute(
                    "SELECT season, number, title, description, summary FROM episodes WHERE doc_id = ?", 
                    (doc_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    enriched_results.append({
                        "doc_id": doc_id,
                        "season": row[0],
                        "number": row[1],
                        "title": row[2],
                        "description": row[3],
                        "summary": row[4]
                    })
                else:
                    # Fallback structural object in case metadata is missing
                    enriched_results.append({
                        "doc_id": doc_id,
                        "season": 0,
                        "number": 0,
                        "title": f"Document {doc_id}",
                        "description": "Metadata missing from database corpus.",
                        "summary": "Summary missing from database corpus."
                    })

        return jsonify({
            "query": query,
            "method": method,
            "count": len(enriched_results),
            "results": enriched_results
        })
    
    except Exception as e:
        logger.exception("search failed")
        return jsonify({"error": "internal error"}), 500


@app.route('/api/rag', methods=['POST'])
def rag():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "query required"}), 400

    if len(query) < 2:
        return jsonify({"error": "query too short"}), 400

    try:
        result = rag_query(query)
        raw_docs = result.get("retrieved_docs", [])

        enriched_docs = []
        with get_db() as conn:
            cursor = conn.cursor()
            for doc in raw_docs:
                # --- SAFELY EXTRACT KEY ---
                # Check for 'doc_id', fall back to 'title' if that's where the ID is stored
                if isinstance(doc, dict):
                    doc_id = doc.get("doc_id") or doc.get("title")
                else:
                    doc_id = str(doc)
                
                cursor.execute(
                    "SELECT season, number, title, description, summary FROM episodes WHERE doc_id = ?", 
                    (doc_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    enriched_docs.append({
                        "doc_id": doc_id,
                        "season": row[0],
                        "number": row[1],
                        "title": row[2],
                        "description": row[3],
                        "summary": row[4]
                    })
                else:
                    # Fallback structurally if SQLite still can't find this doc_id
                    enriched_docs.append({
                        "doc_id": doc_id,
                        "season": 0,
                        "number": 0,
                        "title": f"Document {doc_id}",
                        "description": "Retrieved by RAG, but metadata missing from SQLite.",
                        "summary": "Summary missing from database corpus."
                    })

        return jsonify({
            "query": query,
            "answer": result.get("answer", ""),
            "retrieved_docs": enriched_docs,
            "error": result.get("error"),
            "has_answer": bool(result.get("answer")) and not result.get("error")
        })

    except Exception:
        logger.exception("rag failed")
        return jsonify({"error": "internal error", "has_answer": False}), 500


@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory('static', path)


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "not found"}), 404


@app.errorhandler(500)
def server_error(_):
    return jsonify({"error": "server error"}), 500


if __name__ == '__main__':
    logger.info("Starting IR system")
    app.run(host='0.0.0.0', port=5001, debug=True)
