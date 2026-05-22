# Doctor Who Information Retrieval System

A comprehensive information retrieval system for the **Doctor Who** TV series. This project implements and compares multiple search methods including **Boolean search**, **BM25**, **semantic search with Sentence Transformers**, **FAISS-based nearest neighbor search**, and a **fused search approach** combining sparse and dense retrieval.

## Features

* **Corpus Creation**
  * Loads episode metadata from CSV files (`all-detailsepisodes.csv`, `imdb_details.csv`, `dwguide.csv`)
  * Preprocesses text with tokenization, optional stopword removal, and stemming
  * Builds a **document corpus**, **inverted index**, and **SQLite database**
  * Creates embeddings using `SentenceTransformers` for semantic search

* **Search Methods**
  * **Boolean Search** - Exact matching using an inverted index
  * **BM25 Search** - Probabilistic ranking using `rank_bm25` with configurable k1 and b parameters
  * **Semantic Search** - Dense vector-based retrieval from SQLite embeddings
  * **FAISS Semantic Search** - Fast approximate nearest neighbor search using HNSW index
  * **Fused Search** - Combines BM25 and semantic results using Reciprocal Rank Fusion (RRF)

* **Evaluation Metrics**
  * Computes **P@5** (Precision@5), **R@5** (Recall@5), **AP** (Average Precision)
  * Calculates **MRR** (Mean Reciprocal Rank) and **nDCG** (normalized Discounted Cumulative Gain)
  * Uses gold query-answer pairs for evaluation
  * Supports grid search for BM25 parameter optimization

* **Storage & Persistence**
  * **JSON Files**: `document_corpus_dw.json`, `inverted_index.json`
  * **SQLite Database**: `doctor_who.db` stores episodes, inverted index, and embeddings
  * **FAISS Index**: `faiss.index` with `faiss_mapping.json` for fast semantic retrieval
  * **Results**: CSV outputs for evaluation metrics and parameter sweep results

* **Development Tools**
  * Unit tests for preprocessing, search methods, and evaluation
  * Visualization script for comparing search method performance
  * Parameter tuning capabilities
  * `uv`-based dependency management for reproducibility

## Project Structure

```
doctor-who-ir-project/
│
├─ src/
│   ├─ __init__.py
│   ├─ preprocessing.py          # Text tokenization, stemming, stopword removal
│   ├─ creating_corpus.py        # Corpus & database building pipeline
│   ├─ boolean_search.py         # Inverted index-based exact matching
│   ├─ bm_25.py                  # BM25 probabilistic ranking
│   ├─ semantic_search.py        # Dense embeddings from SQLite
│   ├─ evaluation.py             # Metrics computation (P@5, R@5, AP, MRR, nDCG)
│   ├─ kg_exp.py                 # Knowledge graph experiments
│   └─ kg_exp_chatgpt.py         # KG utilities
│
├─ dw_data/                       # Data directory (auto-created)
│   ├─ all-detailsepisodes.csv   # Episode details
│   ├─ imdb_details.csv          # IMDB episode data
│   ├─ dwguide.csv               # DW guide data
│   ├─ merged_dataset.csv        # Pre-merged episode data
│   ├─ doctor_who.db             # SQLite database (episodes, index, embeddings)
│   ├─ document_corpus_dw.json   # Serialized document corpus
│   ├─ inverted_index.json       # Inverted index for boolean search
│   ├─ faiss.index               # FAISS HNSW index
│   ├─ faiss_mapping.json        # Maps FAISS indices to document IDs
│   ├─ search_results_summary.csv# Evaluation results across all methods
│   ├─ bm25_testing.csv          # BM25 parameter sweep results
│   └─ second_example_20_queries.json # Test queries and gold answers
│
├─ main.py                        # Main evaluation script
├─ compare_methods.py             # Visualization of search method comparison
├─ config.py                      # Centralized configuration
├─ tests.py                       # Unit tests
├─ pyproject.toml                 # Project metadata & dependencies
├─ requirements.txt               # pip dependencies
├─ README.md                      # This file
└─ uv.lock                        # Locked dependency versions
```

## Installation

### Using `uv` (Recommended)

Install `uv` if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the project:

```bash
uv sync
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

### Using `pip` (Alternative)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Build the Corpus and Database

Create the document corpus, inverted index, SQLite database, and FAISS embeddings:

```bash
uv run python src/creating_corpus.py
```

This will:
- Load episode data from CSV files (or use pre-merged `merged_dataset.csv`)
- Preprocess text with tokenization, stemming, and stopword removal
- Create `document_corpus_dw.json` and `inverted_index.json`
- Build `doctor_who.db` with episodes and embeddings
- Generate FAISS index (`faiss.index`) for semantic search

### 2. Run Full Evaluation

Evaluate all search methods on the test queries:

```bash
uv run python main.py
```

Evaluates:
- Boolean Search
- BM25 Search
- Semantic Search
- FAISS Semantic Search
- Fused Search (BM25 + Semantic with RRF)

Results are saved to `dw_data/search_results_summary.csv` with metrics for each query and method.

### 3. Optimize BM25 Parameters

Run a grid search over BM25 `k1` and `b` parameters:

```bash
uv run python main.py --sweep-bm25 --k1-values 0.3 0.4 0.5 0.6 0.7 --b-values 0.7 0.75 0.8 0.85 0.9 1.0 --output bm25_param_tuning.csv
```

Saves aggregated metrics (mean P@5, R@5, MAP, MRR, nDCG) for each parameter combination.

### 4. Compare Search Methods

Generate a visualization comparing all methods:

```bash
uv run python compare_methods.py
```

Creates `dw_data/method_comparison.png` with bar charts for P@5, R@5, AP, MRR, and nDCG.

### 5. Run Tests

Execute unit tests for preprocessing, search methods, and evaluation:

```bash
pytest tests.py -v
```

Tests verify:
- Text preprocessing (tokenization, stemming, stopword removal)
- Boolean and BM25 search functionality
- Evaluation metrics computation

## Configuration

All settings are centralized in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `ENABLE_STOPWORD_REMOVAL` | `True` | Remove common English stopwords |
| `ENABLE_STEMMING` | `True` | Apply Porter stemming |
| `DEFAULT_TOP_K` | `5` | Default number of results |
| `MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence Transformers model |
| `FAISS_M` | `32` | HNSW parameter (max connections) |
| `FAISS_EF_SEARCH` | `50` | HNSW search-time parameter |
| `EXCLUDE_SEASONS` | `["11"]` | Seasons to exclude from corpus |

Modify `config.py` to customize preprocessing, search behavior, or model selection.

## Key Components

### src/preprocessing.py
- **`preprocess_text(text)`**: Tokenizes, stems, and removes stopwords
- Handles `None` and empty inputs gracefully
- Returns list of processed tokens

### src/creating_corpus.py
- **`load_episode_data()`**: Loads from merged CSV or individual source files
- **`build_inverted_index()`**: Creates token→document mapping
- **`build_corpus()`**: Constructs searchable document corpus
- **`build_sqlite_db()`**: Stores episodes and embeddings
- **`build_faiss_index()`**: Creates HNSW index for fast semantic search

### src/bm_25.py
- **`bm25_search_sqlite(query, conn, k1=1.5, b=0.75, top_n=5)`**: BM25 ranking
- Configurable k1 and b parameters for tuning
- Extracts results from SQLite

### src/semantic_search.py
- **`semantic_search_sqlite(query, conn, top_n=5)`**: Dense retrieval from embeddings
- Loads embeddings from SQLite, computes similarities
- Returns top-k by cosine similarity

### src/boolean_search.py
- **`boolean_search_sqlite(query, conn, top_n=5)`**: Exact token matching
- Uses inverted index for fast retrieval

### src/evaluation.py
- **`compute_metrics(retrieved, relevant, top_k=5)`**: Computes:
  - **P@k**: Precision at k
  - **R@k**: Recall at k
  - **AP**: Average Precision
  - **MRR**: Mean Reciprocal Rank
  - **nDCG**: Normalized Discounted Cumulative Gain with graded relevance

### main.py
- **`evaluate_method(name, query_fn)`**: Evaluates a search method
- **`fused_query(...)`**: Combines BM25 and semantic with RRF (k=60)
- **`bm25_param_sweep(...)`**: Grid search over k1 and b

### compare_methods.py
- Loads evaluation results from CSV
- Generates comparison bar charts and metrics table
- Saves visualization to PNG

## Dataset

The corpus is built from three sources:
- **IMDB**: `imdb_details.csv` (episode title, description, season)
- **Episode Details**: `all-detailsepisodes.csv` (additional metadata)
- **DW Guide**: `dwguide.csv` (fan guide summaries)

Episodes are merged on title field. Season 11 is excluded by default (see `config.py`).

Test queries and gold answers are in `dw_data/second_example_20_queries.json`.

## Performance Notes

- **Boolean Search**: Exact matching, fastest but limited recall
- **BM25**: Good balance of precision and recall; best for term-based queries
- **Semantic Search**: Best for conceptual/meaning-based queries; slower than BM25
- **FAISS**: Fastest semantic search (~HNSW approximate nearest neighbors)
- **Fused Search**: Combines strengths of sparse and dense; highest overall performance

Run `compare_methods.py` after evaluation to see detailed comparisons.

## Development

- **Tests**: `pytest tests.py -v` for comprehensive test suite
- **Logging**: Check `app.log` for detailed execution logs
- **Virtual Environment**: Use `uv` for reproducible dependency versions

