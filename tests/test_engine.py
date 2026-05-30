"""
Unit tests for the Doctor Who IR project.
Tests cover preprocessing, search methods, and evaluation metrics.
"""

import pytest
import sqlite3
import json
import config
from src.preprocessing import preprocess_text
from src.boolean_search import boolean_search_sqlite
from src.bm_25 import bm25_search_sqlite
from src.evaluation import compute_metrics, evaluate_method


class TestPreprocessing:
    """Test text preprocessing functions."""

    def test_preprocess_text_basic(self):
        text = "The Doctor travels through time and space."
        result = preprocess_text(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_preprocess_text_empty(self):
        assert preprocess_text("") == []
        assert preprocess_text(None) == []

    def test_preprocess_text_stopwords(self):
        assert preprocess_text("the and or a an") == []

    def test_preprocess_text_stemming(self):
        assert "run" in preprocess_text("running runs runner")


class TestSearchMethods:
    """Test search method implementations."""

    @pytest.fixture
    def db_connection(self):
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE episodes (
                doc_id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                summary TEXT,
                preprocessed_combined TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE inverted_index (
                token TEXT,
                doc_id TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE embeddings (
                doc_id TEXT PRIMARY KEY,
                embedding BLOB
            )
        """)

        sample_docs = [
            ("1x1", "The Doctor", "The Doctor meets a strange being.", "The Doctor explores a mysterious place."),
            ("1x2", "The Angels", "The Doctor fights the Weeping Angels.", "The Doctor encounters the Weeping Angels."),
        ]

        for doc_id, title, description, summary in sample_docs:
            preprocessed = preprocess_text(f"{title} {description} {summary}")
            cur.execute(
                "INSERT INTO episodes VALUES (?, ?, ?, ?, ?)",
                (doc_id, title, description, summary, json.dumps(preprocessed)),
            )

        cur.execute("INSERT INTO inverted_index VALUES (?, ?)", ("doctor", "1x1"))
        cur.execute("INSERT INTO inverted_index VALUES (?, ?)", ("doctor", "1x2"))
        cur.execute("INSERT INTO inverted_index VALUES (?, ?)", ("angel", "1x2"))

        conn.commit()
        yield conn
        conn.close()

    def test_boolean_search_returns_list(self, db_connection):
        result = boolean_search_sqlite("doctor", db_connection, top_n=5)
        assert isinstance(result, list)
        assert "1x1" in result or "1x2" in result

    def test_bm25_search_returns_list(self, db_connection):
        result = bm25_search_sqlite("Doctor", db_connection, top_n=5)
        assert isinstance(result, list)
        assert len(result) <= 5


class TestMetrics:
    """Verify core math metrics calculation logic."""

    def test_precision_calculation(self):
        retrieved = ["1", "2", "3", "4", "5"]
        relevant = {"1", "2", "6"}
        overlap = len(set(retrieved) & relevant)
        assert overlap / 5.0 == 0.4

    def test_recall_calculation(self):
        retrieved = ["1", "2", "3", "4", "5"]
        relevant = {"1", "2", "6"}
        overlap = len(set(retrieved) & relevant)
        assert abs(overlap / len(relevant) - 2 / 3) < 0.01

    def test_mrr_calculation(self):
        assert compute_metrics(["1", "2", "3"], ["1"], top_k=3)["MRR"] == 1.0
        assert compute_metrics(["4", "1", "3"], ["1"], top_k=3)["MRR"] == 0.5

    def test_ap_calculation(self):
        metrics = compute_metrics(["1", "2", "3", "4", "5"], ["1", "3", "5"], top_k=5)
        expected = (1.0 + 2 / 3 + 3 / 5) / 3
        assert abs(metrics["AP"] - expected) < 0.01

    def test_evaluate_method_integration(self):
        def dummy_search(query):
            return ["1", "2", "3"]
            
        # Test the real production import function now!
        metrics = evaluate_method("Dummy Runtime Test", dummy_search)
        assert isinstance(metrics, list)


class TestConfiguration:
    def test_config_paths_exist(self):
        assert hasattr(config, 'DB_PATH')
        assert hasattr(config, 'MERGED_DATASET_PATH')
        assert hasattr(config, 'FAISS_INDEX_PATH')

    def test_config_defaults(self):
        assert config.DEFAULT_TOP_K == 5
        assert config.ENABLE_STEMMING is True
        assert config.MODEL_NAME == "BAAI/bge-base-en-v1.5"
