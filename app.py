"""
Flask web application for Doctor Who Information Retrieval System.
Provides REST API and web UI for RAG and search functionality.
"""

import sqlite3
import logging
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
import faiss
from sentence_transformers import SentenceTransformer

import config
from src.bm_25 import bm25_search_sqlite
from src.boolean_search import boolean_search_sqlite
from src.semantic_search import semantic_search_sqlite
from src.rag import rag_query, format_rag_output

# Configure Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['JSON_SORT_KEYS'] = False

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

# Cache for FAISS index and model
_faiss_index = None
_faiss_mapping = None
_faiss_model = None


def load_faiss_resources():
    """Load FAISS index and model (cached)."""
    global _faiss_index, _faiss_mapping, _faiss_model
    
    if _faiss_index is not None:
        return _faiss_index, _faiss_mapping, _faiss_model
    
    try:
        _faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        _faiss_index.hnsw.efSearch = config.FAISS_EF_SEARCH
        
        with open(config.FAISS_MAPPING_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _faiss_mapping = {int(k): v for k, v in data.items()}
        
        _faiss_model = SentenceTransformer(config.MODEL_NAME)
        
        logger.info("FAISS resources loaded successfully")
        return _faiss_index, _faiss_mapping, _faiss_model
    except Exception as e:
        logger.error(f"Failed to load FAISS resources: {e}")
        return None, None, None


def faiss_search(query, top_k=config.DEFAULT_TOP_K):
    """Perform FAISS search."""
    index, mapping, model = load_faiss_resources()
    if index is None or mapping is None or model is None:
        return []
    
    try:
        query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        distances, indices = index.search(query_emb.reshape(1, -1), top_k)
        return [mapping[i] for i in indices[0] if int(i) in mapping]
    except Exception as e:
        logger.error(f"FAISS search error: {e}")
        return []


def fused_search(query, conn, top_k=config.DEFAULT_TOP_K, k=60, candidate_k=50):
    """Perform fused search combining BM25 and semantic."""
    try:
        bm25_results = bm25_search_sqlite(query, conn, top_n=candidate_k)
        semantic_results = semantic_search_sqlite(query, conn, top_n=candidate_k)
        
        bm25_ranks = {doc: rank + 1 for rank, doc in enumerate(bm25_results)}
        semantic_ranks = {doc: rank + 1 for rank, doc in enumerate(semantic_results)}
        
        fused_scores = {}
        all_docs = set(bm25_results) | set(semantic_results)
        for doc in all_docs:
            rrf_sparse = 1 / (k + bm25_ranks.get(doc, candidate_k + 1))
            rrf_dense = 1 / (k + semantic_ranks.get(doc, candidate_k + 1))
            fused_scores[doc] = rrf_sparse + rrf_dense
        
        sorted_docs = sorted(fused_scores, key=fused_scores.get, reverse=True)
        return sorted_docs[:top_k]
    except Exception as e:
        logger.error(f"Fused search error: {e}")
        return []


@app.route('/')
def index():
    """Serve main page."""
    return render_template('index.html')


@app.route('/api/methods')
def get_methods():
    """Get available search methods."""
    methods = [
        {
            'id': 'boolean',
            'name': 'Boolean Search',
            'description': 'Exact token matching using inverted index',
            'speed': 'Very Fast',
            'metric': 'P@5: 0.27'
        },
        {
            'id': 'bm25',
            'name': 'BM25 Search',
            'description': 'Probabilistic ranking using term frequency and document length',
            'speed': 'Fast',
            'metric': 'P@5: 0.23'
        },
        {
            'id': 'semantic',
            'name': 'Semantic Search',
            'description': 'Dense vector similarity using SentenceTransformers',
            'speed': 'Medium',
            'metric': 'P@5: 0.25'
        },
        {
            'id': 'faiss',
            'name': 'FAISS Search',
            'description': 'Fast approximate nearest neighbor search using HNSW index',
            'speed': 'Very Fast',
            'metric': 'P@5: 0.25'
        },
        {
            'id': 'fused',
            'name': 'Fused Search',
            'description': 'Combines BM25 and semantic results using Reciprocal Rank Fusion',
            'speed': 'Medium',
            'metric': 'P@5: 0.24'
        },
        {
            'id': 'rag',
            'name': 'RAG (AI Generated)',
            'description': 'Retrieves documents and generates answer using Ollama LLM',
            'speed': 'Slow',
            'metric': 'Requires Ollama'
        }
    ]
    return jsonify(methods)


@app.route('/api/search', methods=['POST'])
def search():
    """API endpoint for search."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        method = data.get('method', 'boolean')
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        if len(query) < 2:
            return jsonify({'error': 'Query must be at least 2 characters'}), 400
        
        logger.info(f"Search: query='{query}', method='{method}'")
        
        # RAG is handled separately
        if method == 'rag':
            return jsonify({'error': 'Use /api/rag endpoint for RAG queries'}), 400
        
        # Connect to database
        conn = sqlite3.connect(str(config.DB_PATH))
        
        # Route to appropriate search method
        if method == 'boolean':
            results = boolean_search_sqlite(query, conn, top_n=config.DEFAULT_TOP_K)
        elif method == 'bm25':
            results = bm25_search_sqlite(query, conn, top_n=config.DEFAULT_TOP_K)
        elif method == 'semantic':
            results = semantic_search_sqlite(query, conn, top_n=config.DEFAULT_TOP_K)
        elif method == 'faiss':
            results = faiss_search(query, config.DEFAULT_TOP_K)
        elif method == 'fused':
            results = fused_search(query, conn, config.DEFAULT_TOP_K)
        else:
            conn.close()
            return jsonify({'error': f'Unknown search method: {method}'}), 400
        
        conn.close()
        
        return jsonify({
            'query': query,
            'method': method,
            'results': results,
            'count': len(results)
        })
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rag', methods=['POST'])
def rag():
    """API endpoint for RAG queries."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        if len(query) < 2:
            return jsonify({'error': 'Query must be at least 2 characters'}), 400
        
        logger.info(f"RAG query: '{query}'")
        
        result = rag_query(query)
        
        # Format result for API response
        return jsonify({
            'query': result['query'],
            'retrieved_docs': [doc.get('title', doc) if isinstance(doc, dict) else doc 
                               for doc in result['retrieved_docs']],
            'answer': result['answer'],
            'error': result['error'],
            'has_answer': result['answer'] != '' and result['error'] is None
        })
    
    except Exception as e:
        logger.error(f"RAG error: {e}")
        return jsonify({
            'query': data.get('query', ''),
            'error': str(e),
            'has_answer': False
        }), 500


@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files."""
    return send_from_directory('static', path)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    logger.error(f"Server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("Starting Doctor Who IR Web Application...")
    logger.info("Access at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
