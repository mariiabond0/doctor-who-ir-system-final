"""
Unit tests for the Doctor Who IR project.

Tests cover preprocessing, search methods, and evaluation metrics.
"""

import pytest
import sqlite3
from src.preprocessing import preprocess_text
from src.boolean_search import boolean_search_sqlite
from src.bm_25 import bm25_search_sqlite
from src.evaluation import compute_metrics
import config
import json


class TestPreprocessing:
    """Test text preprocessing functions."""

    def test_preprocess_text_basic(self):
        text = "The Doctor travels through time and space."
        result = preprocess_text(text)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(token.islower() or not token.isalpha() for token in result)

    def test_preprocess_text_empty(self):
        result = preprocess_text("")
        assert result == []

        result = preprocess_text(None)
        assert result == []

    def test_preprocess_text_stopwords(self):
        text = "the and or a an"
        result = preprocess_text(text)
        assert result == []

    def test_preprocess_text_stemming(self):
        text = "running runs runner"
        result = preprocess_text(text)
        assert "run" in result


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
    def test_precision_calculation(self):
        retrieved = ["1", "2", "3", "4", "5"]
        relevant = {"1", "2", "6"}
        overlap = len(set(retrieved) & relevant)
        p_at_5 = overlap / 5.0
        assert p_at_5 == 0.4

    def test_recall_calculation(self):
        retrieved = ["1", "2", "3", "4", "5"]
        relevant = {"1", "2", "6"}
        overlap = len(set(retrieved) & relevant)
        r_at_5 = overlap / len(relevant) if relevant else 0.0
        assert abs(r_at_5 - 2 / 3) < 0.01

    def test_mrr_calculation(self):
        results = ["1", "2", "3"]
        relevant = {"1"}
        metrics = compute_metrics(results, list(relevant), top_k=3)
        assert metrics["MRR"] == 1.0

        results = ["4", "1", "3"]
        metrics = compute_metrics(results, list(relevant), top_k=3)
        assert metrics["MRR"] == 0.5

    def test_ap_calculation(self):
        results = ["1", "2", "3", "4", "5"]
        relevant = {"1", "3", "5"}
        metrics = compute_metrics(results, list(relevant), top_k=5)
        expected = (1.0 + 2 / 3 + 3 / 5) / 3
        assert abs(metrics["AP"] - expected) < 0.01

    def test_evaluate_method(self):
        def dummy_search(query):
            return ["1", "2", "3"]

        def evaluate_method(name, query_fn, queries, answers, top_k=5):
            metrics = []
            for i, query in enumerate(queries):
                result = compute_metrics(query_fn(query), answers[i], top_k=top_k)
                metrics.append(result)
            mean_p5 = sum(m["P@5"] for m in metrics) / len(metrics)
            mean_r5 = sum(m["R@5"] for m in metrics) / len(metrics)
            mean_ap = sum(m["AP"] for m in metrics) / len(metrics)
            mean_mrr = sum(m["MRR"] for m in metrics) / len(metrics)
            return mean_p5, mean_r5, mean_ap, mean_mrr, metrics

        mean_p5, mean_r5, map_score, mean_mrr, _ = evaluate_method(
            "dummy", dummy_search, ["query"], [["1", "2"]], top_k=3
        )
        assert abs(mean_p5 - 2 / 3) < 0.01
        assert abs(mean_r5 - 1.0) < 0.01
        assert mean_mrr == 1.0


class TestConfiguration:
    def test_config_paths_exist(self):
        assert hasattr(config, 'DB_PATH')
        assert hasattr(config, 'CORPUS_PATH')
        assert hasattr(config, 'INDEX_PATH')

    def test_config_defaults(self):
        assert config.DEFAULT_TOP_K == 5
        assert config.ENABLE_STEMMING is True
        assert config.MODEL_NAME == "all-MiniLM-L6-v2"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
