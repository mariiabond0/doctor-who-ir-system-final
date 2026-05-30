# Doctor Who Information Retrieval System

A comprehensive information retrieval system for the **Doctor Who** TV series. This project implements and compares multiple search methods including **Boolean search**, **BM25**, **semantic search with Sentence Transformers**, **FAISS-based nearest neighbor search**, **fused search approach** combining sparse and dense retrieval and **RAG**.

## Features

* **Corpus Creation**
  * Loads episode metadata from CSV files (`all-detailsepisodes.csv`, `imdb_details.csv`, `dwguide.csv`)
  * Preprocesses text with tokenization, optional stopword removal and stemming
  * Builds a **document corpus**, **inverted index**, **FAISS index and mapping** and **SQLite database**
  * Creates embeddings using `SentenceTransformers` for semantic search

* **Search Methods**
  * **Boolean Search** - Exact matching using an inverted index
  * **BM25 Search** - Probabilistic ranking using `rank_bm25` with configurable k1 and b parameters
  * **Semantic Search** - Dense vector-based retrieval from SQLite embeddings
  * **FAISS Semantic Search** - Fast approximate nearest neighbor search using HNSW index
  * **Fused Search** - Combines BM25 and semantic results using Reciprocal Rank Fusion (RRF) followed by deep-learning neural reranking via a Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2).

* **RAG (Retrieval Augmented Generation)**
  * Integrates local **Ollama LLM** for generating answers
  * Retrieves relevant episodes and uses them as context for answer generation
  * Requires `llama3` model or compatible Ollama model running locally

* **Evaluation Metrics**
  * Computes **P@5** (Precision@5), **R@5** (Recall@5), **AP** (Average Precision)
  * Calculates **MRR** (Mean Reciprocal Rank) and **nDCG** (normalized Discounted Cumulative Gain)
  * Uses gold query-answer pairs for evaluation
  * Supports grid search for BM25 parameter optimization via a CLI sweep routine

* **Storage & Persistence**
  * **Data Files (dw_data/)**: SQLite database ('doctor_who.db'), FAISS Index & Mapping ('faiss.index, faiss_mapping.json'), mapping rules, and baseline datasets
  * **Outputs (outputs/)**: Generated structures ('document_corpus_dw.json', 'inverted_index.json'), parameter sweep results, metrics tables, and evaluation visualizations

* **Development Tools**
  * Unit tests for preprocessing, search methods, and evaluation metrics
  * Performance plotting script for comparing search methodology records
  * uv-based dependency management for fast reproducibility

## Project Structure

```
doctor-who-ir-project/
│
├─ src/
│   ├─ __init__.py
│   ├─ bm_25.py                  # BM25 probabilistic ranking
│   ├─ boolean_search.py         # Inverted index-based exact matching
│   ├─ creating_corpus.py        # Corpus & database building pipeline
│   ├─ evaluation.py             # Metrics computation infrastructure
│   ├─ faiss_search.py           # Vector search implementation via FAISS
│   ├─ fused_search.py           # Hybrid sparse/dense RRF retrieval 
│   ├─ preprocessing.py          # Text tokenization, stemming, stopword removal
│   ├─ rag.py                    # RAG with Ollama LLM integration
│   ├─ semantic_search.py        # Dense embeddings from SQLite
│   └─ kg_builders/              # Knowledge graph generation experiments
│
├─ dw_data/                      # Source datasets & working DB
│   ├─ all-detailsepisodes.csv
│   ├─ all-scripts.csv
│   ├─ doctor_who.db             # Main SQLite storage
│   ├─ dwguide.csv
│   ├─ faiss_mapping.json
│   ├─ faiss.index
│   ├─ imdb_details.csv
│   ├─ merged_dataset.csv
│   └─ second_example_20_queries.json
│
├─ outputs/                      # Generated evaluation artifacts
│   ├─ bm25_param_tuning.csv
│   ├─ document_corpus_dw.json
│   ├─ inverted_index.json
│   ├─ method_comparison.png
│   ├─ method_ranking.csv
│   └─ search_results_summary.csv
│
├─ graph_depo/                   # Knowledge Graph files
│
├─ static/                       # Web frontend styling/logic
│   ├─ css/
│   └─ js/
│
├─ templates/
│   └─ index.html                # Main interface template
│
├─ tests/                        # Application testing framework
│   ├─ __init__.py
│   └─ test_engine.py
│
├─ app.py                        # Flask Web Application entry point
├─ app.log
├─ compare_methods.py            # Search method visualization
├─ config.py                     # Centralized configuration
├─ main.py                       # CLI Evaluation and execution entry point
├─ requirements.txt              # pip dependencies
├─ pyproject.toml                # Project metadata & dependencies
├─ README.md                     # This file
└─ uv.lock                       # Locked dependency versions
```

## Installation

### Using `uv` (Recommended)

Install `uv` if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the project and sync dependencies:

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

Create the document corpus, inverted index, SQLite database tables, and FAISS embeddings:

```bash
uv run python src/creating_corpus.py
```

This will:
- Load episode data from CSV files (or use pre-merged `merged_dataset.csv`)
- Preprocess text with tokenization, stemming, and stopword removal
- Create `document_corpus_dw.json` and `inverted_index.json`
- Build `doctor_who.db` with episodes and embeddings
- Generate FAISS index (`faiss.index`) for semantic search

### 2. Launch Web UI (Recommended)

Start the Flask web application on port 5001:

```bash
uv run python app.py
```

Then open your browser to **http://localhost:5001**

**Features:**
- 🎯 Unified search interface with method selector
- 🤖 RAG mode for AI-generated answers (requires Ollama)
- 📊 Real-time results with episode cards
- ⚡ Fast, clean, modern UI

### 3. Run Evaluation Mode

Evaluate all search methods on your test queries via the CLI:

```bash
uv run python main.py --mode eval
```

Evaluates:
- Boolean Search
- BM25 Search
- Semantic Search
- FAISS Semantic Search
- Fused Search (BM25 + Semantic with RRF)

Results are saved to `outputs/search_results_summary.csv`.

ℹ️ Note: The first time you execute an evaluation or search query using the fused pipeline, the application will automatically pull the cross-encoder reranker model (cross-encoder/ms-marco-MiniLM-L-6-v2) from Hugging Face. This requires an active internet connection.

### 4. Run CLISearch Modes

**Search with specific method:**
```bash
uv run python main.py --mode search --query "Daleks" --search-method boolean
```
(Available choices: boolean, bm25, semantic, faiss, fused)

**RAG mode (CLI answering):**
```bash
uv run python main.py --mode rag --query "Who is the Doctor?"
```

### 5. Optimize BM25 Parameters (Grid Search)

Run a parameter sweep grid directly across multiple values of k1 and b. This executes the optimization pipeline defined inside src/evaluation.py:

```bash
uv run python main.py --sweep-bm25 --output bm25_param_tuning.csv
```

### 6. Compare Search Methods

Generate a visualization plot comparing all metrics across all methods:

```bash
uv run python compare_methods.py
```

Creates `outputs/method_comparison.png` with bar charts for P@5, R@5, AP, MRR, and nDCG.

### 7. Run Verification Tests

Run the full testing framework suite through pytest:

```bash
uv run pytest tests/test_engine.py -v
```

## Configuration

All configuration variables are centralized in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `ENABLE_STOPWORD_REMOVAL` | `True` | Remove common English stopwords |
| `ENABLE_STEMMING` | `True` | Apply Porter stemming |
| `DEFAULT_TOP_K` | `5` | Default number of results |
| `MODEL_NAME` | `BAAI/bge-base-en-v1.5` | Sentence Transformers model |
| `FAISS_M` | `64` | HNSW parameter (max connections) |
| `FAISS_EF_SEARCH` | `50` | HNSW search-time parameter |
| `EXCLUDE_SEASONS` | `["11"]` | Seasons to exclude from corpus |
| `RAG_ENABLED` | `True` | Enable RAG mode |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server endpoint |
| `OLLAMA_MODEL` | `llama3` | Ollama model to use |
| `RAG_CONTEXT_SIZE` | `5` | Number of retrieved documents for context |
| `RAG_TEMPERATURE` | `0.7` | LLM generation temperature (0.0-1.0) |
| `RAG_MAX_TOKENS` | `500` | Maximum tokens in generated answer |

## Key Components

### main.py
- Core execution router managing parsing arguments for --mode eval, --mode search, --mode rag, and --sweep-bm25.
- Manages runtime instantiation of index handlers and database connections.

### app.py (Flask Web Application)
- **`GET /`**: Serves the main web UI (index.html)
- **`GET /api/methods`**: Returns available search methods with descriptions and performance metrics
- **`POST /api/search`**: Performs search with specified method
  - Parameters: `query` (string), `method` (boolean|bm25|semantic|faiss|fused)
  - Returns: `results` (list of episode IDs), `count`, `method`
- **`POST /api/rag`**: Performs RAG search by dynamically processing, fetching, and formatting structural metadata records from the SQLite database.
- **`GET /static/<path>`**: Serves static files (CSS, JavaScript)

### src/evaluation.py
- **`compute_metrics`**: Computes:
  - **P@k**: Precision at k
  - **R@k**: Recall at k
  - **AP**: Average Precision
  - **MRR**: Mean Reciprocal Rank
  - **nDCG**: Normalized Discounted Cumulative Gain with graded relevance
- **`bm25_param_sweep()`**: Houses the processing logic for tracking metric shifts across experimental hyperparameter spaces.

### src/faiss_search.py
- **`faiss_query(query, index, mapping, model, top_k)`**: Converts incoming text into dense vector queries and extracts fast nearest-neighbors maps from HNSW indices.

### src/fused_search.py
- **`fused_query()`**: Runs multi-stage hybrid search workflows. Combines sparse tables (BM25) with dense indexes via Reciprocal Rank Fusion (RRF) and reranks the results using cross-encoders.

### Web UI Files
- **`templates/index.html`**: Main page with query interface and results display
- **`static/js/app.js`**: Client-side form handling, API calls, results rendering
- **`static/css/style.css`**: Responsive design with Bootstrap 5 integration

## Web API

The Flask application provides a REST API for programmatic access:

### Health Check
```bash
curl http://localhost:5001/api/methods
```

### Search
```bash
curl -X POST http://localhost:5001/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Daleks", "method": "fused"}'
```

### RAG Query
```bash
curl -X POST http://localhost:5001/api/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "Who is the Doctor?"}'
```

## Dataset

The corpus is built from three sources:
- **IMDB**: `imdb_details.csv` (episode title, description, season)
- **Episode Details**: `all-detailsepisodes.csv` (additional metadata)
- **DW Guide**: `dwguide.csv` (fan guide summaries)

Episodes are merged on title field. Season 11 is excluded by default (see `config.py`).

Test queries and gold answers are in `dw_data/second_example_20_queries.json`.

## Performance Notes

- **Fused Search (Hybrid RRF + Reranking)**: Highest overall performance (Normalized Score: 1.000). By combining the lexical precision of BM25 with the conceptual abstraction of dense vectors, it leads across every single metric—notably achieving a Mean MRR of 0.704 and a Mean nDCG of 0.362. Use this when accuracy is the absolute priority.
- **BM25**: While slightly trailing semantic variants in broad coverage (Mean P@5: 0.260 vs 0.270), it significantly outperforms pure semantic search on position-weighted metrics, capturing a Mean MRR of 0.617 (compared to Semantic's 0.571). It remains the best baseline for exact keyword/entity queries (e.g., specific alien races or episode titles).
- **FAISS Semantic Search vs. Vanilla Semantic**: Identical retrieval quality with massive speed gains. Both methods yield identical precision and recall profiles (Mean P@5: 0.270, Mean R@5: 0.277), but FAISS utilizes a localized HNSW approximate nearest-neighbor index to bypass exhaustive database cosine-similarity loops, offering significantly lower latency.
- **Boolean Search**: Exact keyword intersection provides the lowest computational overhead but suffers from severe vocabulary mismatch, resulting in the lowest metrics across the board (Mean AP: 0.153).