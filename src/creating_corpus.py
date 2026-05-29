import json
import os

# import pickle
import logging
import sqlite3
from collections import defaultdict

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import config
from src.preprocessing import preprocess_text

LOGGER = logging.getLogger(__name__)
MODEL_NAME = config.MODEL_NAME


def load_episode_data() -> pd.DataFrame:
    """Load episode metadata.

    Prefer a pre-merged CSV at `config.MERGED_DATASET_PATH`. If it doesn't exist,
    attempt to read the individual source CSVs and create the merged file. If
    required source files are missing, raise a clear error.
    """
    # If a pre-merged dataset is available, use it (this is the common case).
    if config.MERGED_DATASET_PATH.exists():
        return pd.read_csv(config.MERGED_DATASET_PATH)

    # Fallback: try to build the merged dataset from source files.
    try:
        df_details = pd.read_csv(config.EPISODES_CSV)
        df_imdb = pd.read_csv(config.IMDB_CSV)
        df_guide = pd.read_csv(config.DW_GUIDE_CSV)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing required CSV file: {exc.filename}.\nEither provide {config.MERGED_DATASET_PATH} or ensure {config.EPISODES_CSV}, {config.IMDB_CSV}, and {config.DW_GUIDE_CSV} exist."
        ) from exc
    #merged = pd.merge(
        df_imdb[["number", "title", "description", "season"]],
        df_details["title"],
        df_guide[["title", "summary"]],
        on="title",
        how="left",
    #)
    base = df_imdb[["number", "title", "description", "season"]]
    merged = base.merge(
        df_details[["title"]],
        on="title",
        how="left",
        suffixes=("", "_details")
    )
    merged = merged.merge(
        df_guide[["title", "summary"]],
        on="title",
        how="left"
    )
    merged.to_csv(config.MERGED_DATASET_PATH, index=False)
    return merged


def filter_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out seasons that are excluded by configuration."""
    if config.EXCLUDE_SEASONS:
        return df[~df["season"].astype(str).isin(config.EXCLUDE_SEASONS)]
    return df


def document_id(season: int, number: int) -> str:
    """Return a canonical document identifier for a season/episode."""
    return f"{season}x{number}"


def build_corpus(df: pd.DataFrame):
    """Build a document corpus, inverted index, and embeddings for the dataset."""
    document_corpus = {}
    inverted_index = defaultdict(set)
    embeddings_dict = {}
    model = SentenceTransformer(MODEL_NAME)

    # for _, row in df.iterrows():
    #    doc_id = document_id(int(row["season"]), int(row["number"]))
    #    title = str(row.get("title", "")).strip()
    #    description = str(row.get("description", "")).strip()
    #    text = f"{title} {description}".strip()

    #    document_corpus[doc_id] = {
    #        "id": doc_id,
    #        "season": int(row["season"]),
    #        "number": int(row["number"]),
    #        "title": title,
    #        "description": description,
    #    }

    for row in df.itertuples(index=False):
        season = int(row.season)
        number = int(row.number)

        doc_id = document_id(season, number)

        title = str(getattr(row, "title", "")).strip()
        description = str(getattr(row, "description", "")).strip()
        summary = str(getattr(row, "summary", "")).strip()
        text = f"{title} {description} {summary}".strip()

        document_corpus[doc_id] = {
            "id": doc_id,
            "season": season,
            "number": number,
            "title": title,
            "description": description,
            "summary": summary,
        }

        preprocessed = preprocess_text(text)
        for token in preprocessed:
            inverted_index[token].add(doc_id)

        embeddings_dict[doc_id] = model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )

    return document_corpus, inverted_index, embeddings_dict


def sort_corpus(document_corpus):
    """Sort corpus entries by season and episode number."""
    return dict(
        sorted(
            document_corpus.items(),
            key=lambda item: (item[1]["season"], item[1]["number"]),
        )
    )


def save_json_corpus(document_corpus, inverted_index):
    """Persist the corpus and inverted index to JSON files."""
    with open(config.CORPUS_PATH, "w", encoding="utf-8") as output:
        json.dump(document_corpus, output, ensure_ascii=False, indent=2)

    inverted_index_json = {
        token: sorted(list(doc_ids)) for token, doc_ids in inverted_index.items()
    }
    with open(config.INDEX_PATH, "w", encoding="utf-8") as output:
        json.dump(inverted_index_json, output, ensure_ascii=False, indent=2)

    LOGGER.info("Corpus JSON and inverted index saved.")


# def save_database(document_corpus, inverted_index, embeddings_dict):
#     """Write the corpus, inverted index, and embeddings to SQLite."""
#     conn = sqlite3.connect(str(config.DB_PATH))
#     cur = conn.cursor()

#     cur.execute(
#         """
#         CREATE TABLE IF NOT EXISTS episodes (
#             doc_id TEXT PRIMARY KEY,
#             season INTEGER,
#             number INTEGER,
#             title TEXT,
#             description TEXT,
#             preprocessed_combined BLOB
#         )
#         """
#     )

#     cur.execute(
#         """
#         CREATE TABLE IF NOT EXISTS inverted_index (
#             token TEXT,
#             doc_id TEXT
#         )
#         """
#     )

#     cur.execute(
#         """
#         CREATE TABLE IF NOT EXISTS embeddings (
#             doc_id TEXT PRIMARY KEY,
#             embedding BLOB
#         )
#         """
#     )

#     cur.execute("DELETE FROM episodes")
#     cur.execute("DELETE FROM inverted_index")
#     cur.execute("DELETE FROM embeddings")

#     for doc_id, doc in document_corpus.items():
#         preprocessed_document = preprocess_text(f"{doc['title']} {doc['description']}")
#         cur.execute(
#             "INSERT OR REPLACE INTO episodes VALUES (?, ?, ?, ?, ?, ?)",
#             (
#                 doc_id,
#                 doc["season"],
#                 doc["number"],
#                 doc["title"],
#                 doc["description"],
#                 pickle.dumps(preprocessed_document),
#             ),
#         )

#     for token, doc_ids in inverted_index.items():
#         for doc_id in sorted(doc_ids):
#             cur.execute("INSERT INTO inverted_index VALUES (?, ?)", (token, doc_id))

#     for doc_id, embedding in embeddings_dict.items():
#         cur.execute(
#             "INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
#             (doc_id, sqlite3.Binary(pickle.dumps(embedding))),
#         )

#     conn.commit()
#     conn.close()
#     LOGGER.info("SQLite database saved.")


def save_database(document_corpus, inverted_index, embeddings_dict, precomputed_tokens=None):
    """Efficiently write corpus, inverted index, and embeddings to SQLite."""

    conn = sqlite3.connect(str(config.DB_PATH))
    cur = conn.cursor()

    # --- SQLite tuning ---
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA foreign_keys = ON;")

    # --- Schema ---
    cur.executescript("""
        DROP TABLE IF EXISTS episodes;
        DROP TABLE IF EXISTS inverted_index;
        DROP TABLE IF EXISTS embeddings;

        CREATE TABLE episodes (
            doc_id TEXT PRIMARY KEY,
            season INTEGER,
            number INTEGER,
            title TEXT,
            description TEXT,
            summary TEXT,
            preprocessed_combined TEXT
        );

        CREATE TABLE inverted_index (
            token TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES episodes(doc_id)
                ON DELETE CASCADE
        );

        CREATE TABLE embeddings (
            doc_id TEXT PRIMARY KEY,
            embedding BLOB,
            FOREIGN KEY(doc_id) REFERENCES episodes(doc_id)
                ON DELETE CASCADE
        );

        CREATE INDEX idx_token ON inverted_index(token);
        CREATE INDEX idx_doc_id ON inverted_index(doc_id);
        """)

    # --- Prepare episodes batch ---
    episodes_rows = []
    for doc_id, doc in document_corpus.items():
        season = int(doc["season"])
        number = int(doc["number"])
        title = (doc.get("title") or "").strip()
        description = (doc.get("description") or "").strip()
        summary = (doc.get("summary") or "").strip()
        preprocessed = preprocess_text(f"{title} {description} {summary}")

        episodes_rows.append(
            (
                doc_id,
                season,
                number,
                title,
                description,
                summary,
                json.dumps(preprocessed, ensure_ascii=False),
            )
        )

    cur.executemany("INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?)", episodes_rows)

    # --- Prepare inverted index batch ---
    inverted_rows = []
    for token, doc_ids in inverted_index.items():
        for doc_id in doc_ids:  # without sorted for speed
            inverted_rows.append((token, doc_id))

    cur.executemany("INSERT INTO inverted_index VALUES (?, ?)", inverted_rows)

    # --- Prepare embeddings batch ---
    embedding_rows = []
    for doc_id, embedding in embeddings_dict.items():
        if isinstance(embedding, np.ndarray):
            emb_bytes = embedding.astype("float32").tobytes()
        else:
            emb_bytes = np.array(embedding, dtype="float32").tobytes()

        embedding_rows.append((doc_id, sqlite3.Binary(emb_bytes)))

    cur.executemany("INSERT INTO embeddings VALUES (?, ?)", embedding_rows)

    conn.commit()
    conn.close()

    LOGGER.info("SQLite database saved (optimized).")


def build_faiss_index(embeddings_dict):
    """Build and save a FAISS index for the corpus embeddings."""
    doc_ids = list(embeddings_dict.keys())
    embedding_matrix = np.vstack([embeddings_dict[doc_id] for doc_id in doc_ids]).astype("float32")

    faiss.normalize_L2(embedding_matrix)
    dimension = embedding_matrix.shape[1]
    os.environ["OMP_NUM_THREADS"] = "1"

    index = faiss.IndexHNSWFlat(dimension, config.FAISS_M)
    index.hnsw.efConstruction = config.FAISS_EF_CONSTRUCTION
    index.hnsw.efSearch = config.FAISS_EF_SEARCH
    index.add(embedding_matrix)

    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    index_to_doc_id = {i: doc_id for i, doc_id in enumerate(doc_ids)}
    with open(config.FAISS_MAPPING_PATH, "w", encoding="utf-8") as mapping_file:
        json.dump(index_to_doc_id, mapping_file, ensure_ascii=False, indent=2)

    LOGGER.info("FAISS index and mapping saved.")


def main():
    logging.basicConfig(level=logging.INFO)
    df = load_episode_data()
    df = filter_seasons(df)
    document_corpus, inverted_index, embeddings_dict = build_corpus(df)
    document_corpus = sort_corpus(document_corpus)

    save_json_corpus(document_corpus, inverted_index)
    save_database(document_corpus, inverted_index, embeddings_dict)
    build_faiss_index(embeddings_dict)

    print(f"Corpus created: {len(document_corpus)} episodes saved.")


if __name__ == "__main__":
    main()
