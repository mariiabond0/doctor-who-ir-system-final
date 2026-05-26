"""
Configuration file for the Doctor Who IR project.
Centralizes all paths, settings, and configuration constants.
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "dw_data"

# Database paths
DB_PATH = DATA_DIR / "doctor_who.db"

# Data file paths
CORPUS_PATH = DATA_DIR / "document_corpus_dw.json"
INDEX_PATH = DATA_DIR / "inverted_index.json"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
FAISS_MAPPING_PATH = DATA_DIR / "faiss_mapping.json"
MERGED_DATASET_PATH = DATA_DIR / "merged_dataset.csv"
RESULTS_CSV_PATH = DATA_DIR / "search_results_summary.csv"
QUERIES_PATH = DATA_DIR / "second_example_20_queries.json"

# Input data files
EPISODES_CSV = DATA_DIR / "all-detailsepisodes.csv"
IMDB_CSV = DATA_DIR / "imdb_details.csv"
DW_GUIDE_CSV = DATA_DIR / "dwguide.csv"

# Preprocessing settings
ENABLE_STOPWORD_REMOVAL = True
ENABLE_STEMMING = True

# Corpus filtering
EXCLUDE_SEASONS = ["11"]  # Seasons to exclude from corpus

# Search settings
DEFAULT_TOP_K = 5
FAISS_EF_CONSTRUCTION = 400  # HNSW parameter (increased from 200)
FAISS_M = 64  # HNSW parameter (increased from 32)
FAISS_EF_SEARCH = 200  # HNSW search-time parameter (increased from 50)

# Model settings
MODEL_NAME = "all-MiniLM-L6-v2"
#MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"

# RAG settings (Retrieval Augmented Generation)
RAG_ENABLED = True
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama2"
RAG_CONTEXT_SIZE = 5  # Number of retrieved documents for context
RAG_TEMPERATURE = 0.7  # Generation temperature (0.0-1.0)
RAG_MAX_TOKENS = 500  # Maximum tokens for answer generation

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "app.log"

# Evaluation settings
EVALUATION_TOP_K = 5

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)
