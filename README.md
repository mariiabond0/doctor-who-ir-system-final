# Doctor Who Information Retrieval System

This project implements an information retrieval system for the **Doctor Who** TV series. It supports multiple search methods, including **Boolean search**, **BM25**, **semantic search with Sentence Transformers**, **FAISS-based nearest neighbor search**, and a **knowledge graph extraction pipeline** using GliNER2.

## Features

* **Corpus Creation**
  * Collects episode metadata from CSV files (`all-detailsepisodes.csv`, `imdb_details.csv`, `dw_guide_details.csv`)
  * Preprocesses text with tokenization, optional stopword removal, and stemming
  * Builds a **document corpus**, **inverted index**, and **SQLite database**
* **Search Methods**
  * **Boolean Search** using an inverted index
  * **BM25 Search** using `rank_bm25`
  * **Semantic Search** using `SentenceTransformers` (`all-MiniLM-L6-v2`)
  * **FAISS Semantic Search** using a nearest neighbor index
  * **Fused Search** combining BM25 and semantic with Reciprocal Rank Fusion (RRF)
* **Evaluation**
  * Uses gold query-answer pairs to compute P@5, R@5, AP, MRR, and nDCG
  * Supports BM25 parameter sweeps for tuning `k1` and `b`
* **Knowledge Graph**
  * Extracts entities and relations using GliNER2
  * Builds a graph from text documents
  * Supports entity linking and graph querying
  * Supports KG-only search and per-query KG result summaries
* **Storage Options**
  * JSON files (`document_corpus_dw.json`, `inverted_index.json`, `knowledge_graph.json`)
  * SQLite database (`doctor_who.db`) with episodes, inverted index, and embeddings
  * FAISS index for fast semantic nearest neighbor retrieval
* **Deployment**
  * `uv`-based dependency management

## Project Structure

```
doctor-who-ir-project/
│
├─ src/
│   ├─ __init__.py
│   ├─ bm_25.py
│   ├─ boolean_search.py
│   ├─ creating_corpus.py
│   ├─ knowledge_graph.py
│   ├─ preprocessing.py
│   ├─ semantic_search.py
│
├─ dw_data/
│   ├─ all-detailsepisodes.csv
│   ├─ all-scripts.csv
│   ├─ doctor_who.db
│   ├─ document_corpus_dw.json
│   ├─ dwguide.csv
│   ├─ first_example_10_queries.json
│   ├─ imdb_details.csv
│   ├─ inverted_index.json
│   ├─ merged_dataset.csv
│   ├─ faiss_mapping.json
│   ├─ faiss.index
│   ├─ bm25_testing.csv
│   ├─ knowledge_graph.json
│   ├─ method_comparison.png
│   └─ search_results_summary.csv
│
├─ main.py
├─ compare_methods.py
├─ README.md
├─ requirements.txt
├─ pyproject.toml
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

1. Build the corpus and database:

```bash
uv run python src/creating_corpus.py
```

2. Run evaluation:

```bash
uv run python main.py
```

If `knowledge_graph.json` exists in `dw_data`, evaluation also includes a separate Knowledge Graph Search step and saves a per-query KG search summary to `dw_data/kg_search_summary.json`.

3. Build a knowledge graph from the episode corpus:

```bash
uv run python main.py --build-kg --kg-model gliner2/gliner2-base --kg-output knowledge_graph.json
```

4. Run KG-only search:

```bash
uv run python main.py --kg-search --kg-output knowledge_graph.json --kg-results-output kg_search_results_summary.csv
```

5. Query the saved knowledge graph by triple pattern:

```bash
uv run python main.py --kg-output knowledge_graph.json --kg-query-subject "Doctor Who"
```

6. Run BM25 parameter sweep:

```bash
uv run python main.py --sweep-bm25 --k1-values 0.3 0.4 0.5 0.6 0.7 --b-values 0.7 0.75 0.8 0.85 0.9 1.0 --output bm25_testing.csv
```

> Note: `--kg-output` and `--kg-results-output` are file names relative to `dw_data/` because the script prefixes them with the data directory.

## Notes

* `config.py` centralizes paths and search settings.
* `src/semantic_search.py` provides semantic nearest-neighbor search from SQLite embeddings.
* `src/knowledge_graph.py` implements KG extraction, entity linking, and JSON persistence.

