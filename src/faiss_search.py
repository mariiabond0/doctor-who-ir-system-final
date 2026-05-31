"""
FAISS HNSW fast semantic retrieval engine for the Doctor Who IR system.
"""

import json
import logging
import faiss
from sentence_transformers import SentenceTransformer
import config

logger = logging.getLogger(__name__)


def load_faiss_mapping(mapping_path):
    with open(mapping_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return {int(k): v for k, v in data.items()}


def faiss_query(query, index, mapping, model, top_k=config.DEFAULT_TOP_K):
    bge_instruction = "Represent this sentence for searching relevant passages: "
    processed_query = bge_instruction + query
    query_emb = model.encode(
        processed_query, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    distances, indices = index.search(query_emb.reshape(1, -1), top_k)
    return [mapping[i] for i in indices[0] if int(i) in mapping]
