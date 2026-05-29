# Doctor Who Information Retrieval System

A comprehensive information retrieval system for the **Doctor Who** TV series. This project implements and compares multiple search methods including **Boolean search**, **BM25**, **semantic search with Sentence Transformers**, **FAISS-based nearest neighbor search**, **fused search approach**, and **RAG** combining sparse and dense retrieval.

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

* **RAG (Retrieval Augmented Generation)**
  * Integrates local **Ollama LLM** for generating contextual answers
  * Retrieves relevant episodes and uses them as context for answer generation
  * Requires `llama3` model or compatible Ollama model running locally

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
│   ├─ rag.py                    # RAG with Ollama LLM integration
│   ├─ evaluation.py             # Metrics computation (P@5, R@5, AP, MRR, nDCG)
│   └─ kg_builders/              # Knowledge graph experiments
│       ├─ kg_exp.py             # by myself
│       ├─ kg_exp_chatgpt.py     # by chatgpt
│       └─ kg_exp_claude.py      # by Claude
│
├─ dw_data/                      # Data directory
│   ├─ all-detailsepisodes.csv   # Episode details
│   ├─ all-scripts.csv           # Episode scripts details
│   ├─ bm25_testing.csv          # BM25 parameter sweep results
│   ├─ doctor_who.db             # SQLite database (episodes, index, embeddings)
│   ├─ document_corpus_dw.json   # Serialized document corpus
│   ├─ dwguide.csv               # DW guide data
│   ├─ faiss_mapping.json        # Maps FAISS indices to document IDs
│   ├─ faiss.index               # FAISS HNSW index
│   ├─ first_example_10_queries  # initial test queries
│   ├─ imdb_details.csv          # IMDB episode data
│   ├─ inverted_index.json       # Inverted index for boolean search
│   ├─ merged_dataset.csv        # Pre-merged episode data
│   ├─ method_comparison.png     
│   ├─ method_ranking.csv
│   ├─ search_results_summary.csv# Evaluation results across all methods
│   └─ second_example_20_queries.json # Test queries and gold answers
│
├─ graph_depo/                   # Knowledge Graphs visualizations
│   ├─ doctor_who_kg_claude.graphml
│   ├─ doctor_who_kg_claude.html
│   ├─ doctor_who_kg.graphml
│   └─ doctor_who_kg.html
│
├─ evaluations_testings/          # Tests and comparisons
│   ├─ compare_methods.py         # Visualization of search method comparison
│   └─ tests.py                   # Unit tests
│
├─ main.py                        # Main evaluation script
├─ config.py                      # Centralized configuration
├─ app.py                         # UI start
├─ pyproject.toml                 # Project metadata & dependencies
├─ requirements.txt               # pip dependencies
├─ README.md                      # This file
├─ app.log
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

### 2. Launch Web UI (Recommended)

Start the Flask web application with a beautiful, responsive interface:

```bash
uv run python app.py
```

Then open your browser to **http://localhost:5001**

**Features:**
- 🎯 Unified search interface with method selector
- 🤖 RAG mode for AI-generated answers (requires Ollama)
- 📊 Real-time results with episode cards
- 📱 Fully responsive design (desktop, tablet, mobile)
- ⚡ Fast, clean, modern UI

**Screenshots:**
- Query box at top with search method dropdown
- Results display as episode cards with ranking
- RAG mode shows retrieved documents + AI answer
- Method information panel with performance metrics

### 3. Run Evaluation Mode

Evaluate all search methods on the test queries (CLI mode):

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

### 4. Run CLI Modes

**Search with specific method:**
```bash
uv run python main.py --mode search --query "Daleks" --search-method boolean
```

**RAG mode (CLI):**
```bash
uv run python main.py --mode rag --query "Who is the Doctor?"
```

### 5. Optimize BM25 Parameters

Run a grid search over BM25 `k1` and `b` parameters:

```bash
uv run python main.py --sweep-bm25 --k1-values 0.3 0.4 0.5 0.6 0.7 --b-values 0.7 0.75 0.8 0.85 0.9 1.0 --output bm25_param_tuning.csv
```

Saves aggregated metrics (mean P@5, R@5, MAP, MRR, nDCG) for each parameter combination.

### 6. Compare Search Methods

Generate a visualization comparing all methods:

```bash
uv run python compare_methods.py
```

Creates `dw_data/method_comparison.png` with bar charts for P@5, R@5, AP, MRR, and nDCG.

### 7. Use RAG for Answer Generation

Generate AI-powered answers using local Ollama LLM (via web UI or CLI):

**Prerequisites:**
1. Install Ollama: https://ollama.ai
2. Pull the model: `ollama pull llama3` (or your preferred model)
3. Start Ollama server: `ollama serve`

**Via Web UI (Recommended):**
1. Start: `uv run python app.py`
2. Go to http://localhost:5001
3. Select method: "RAG - AI Generated Answer"
4. Enter query: "Who is the Doctor?"
5. View AI answer with source documents

> Tip: If running via uv on a different host/port, replace the URL accordingly (e.g., http://127.0.0.1:5001).

**Via CLI:**

```bash
uv run python main.py --mode rag --query "Who is the Doctor?"
```

The system will:
- Retrieve the 5 most relevant Doctor Who episodes
- Pass them as context to Ollama's llama3 model
- Generate a contextual answer based on the retrieved episodes

**Example output:**
```
================================================================================
RAG ANSWER
================================================================================

Question: Who is the Doctor?

Retrieved 5 documents:
  1. 1x1 (score: 0.800)
  2. 2x4 (score: 0.750)
  3. 1x9 (score: 0.725)
  4. 5x1 (score: 0.700)
  5. 7x5 (score: 0.675)

────────────────────────────────────────────────────────────────────────────────
Generated Answer:
────────────────────────────────────────────────────────────────────────────────

The Doctor is a mysterious time traveler from the planet Gallifrey who...
```

### 8. Run Tests

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
| `RAG_ENABLED` | `True` | Enable RAG mode |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server endpoint |
| `OLLAMA_MODEL` | `llama3` | Ollama model to use |
| `RAG_CONTEXT_SIZE` | `5` | Number of retrieved documents for context |
| `RAG_TEMPERATURE` | `0.7` | LLM generation temperature (0.0-1.0) |
| `RAG_MAX_TOKENS` | `500` | Maximum tokens in generated answer |

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

### src/rag.py
- **`rag_query(query)`**: End-to-end RAG pipeline
  - Retrieves top-K episodes using Boolean Search
  - Builds context from retrieved documents
  - Queries Ollama LLM with context
  - Returns structured result with answer and source documents
- **`check_ollama_health()`**: Checks if Ollama server is running
- **`retrieve_context(query, top_k=5)`**: Retrieves and formats episode context
- **`build_prompt(query, context)`**: Constructs LLM prompt with context
- **`query_ollama(prompt, temperature, max_tokens)`**: Sends prompt to Ollama
- **`format_rag_output(result)`**: Formats RAG result for display

### app.py (Flask Web Application)
- **`GET /`**: Serves the main web UI (index.html)
- **`GET /api/methods`**: Returns available search methods with descriptions and performance metrics
- **`POST /api/search`**: Performs search with specified method
  - Parameters: `query` (string), `method` (boolean|bm25|semantic|faiss|fused)
  - Returns: `results` (list of episode IDs), `count`, `method`
- **`POST /api/rag`**: Performs RAG query
  - Parameters: `query` (string)
  - Returns: `answer` (generated text), `retrieved_docs` (source episodes), `error` (if any)
- **`GET /static/<path>`**: Serves static files (CSS, JavaScript)

### Web UI Files
- **`templates/index.html`**: Main page with query interface and results display
- **`static/js/app.js`**: Client-side form handling, API calls, results rendering
- **`static/css/style.css`**: Responsive design with Bootstrap 5 integration

## Web API

The Flask application provides a REST API for programmatic access:

### Health Check
```bash
curl http://localhost:5000/api/methods
```

### Search
```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Daleks", "method": "boolean"}'
```

### RAG Query
```bash
curl -X POST http://localhost:5000/api/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "Who is the Doctor?"}'
```

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

