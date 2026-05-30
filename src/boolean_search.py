import sqlite3

from collections import Counter
from src.preprocessing import preprocess_text

"""Simple Boolean search using inverted index (term frequency ranking)"""


def boolean_search(query: str, index: dict, top_n=5):
    query_tokens = preprocess_text(query)
    if not query_tokens:
        return []

    doc_scores = Counter()

    for token in query_tokens:
        for doc_id in index.get(token, []):
            doc_scores[doc_id] += 1

    # sorting documents by score (number of matching tokens) in descending order
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    return [doc_id for doc_id, _ in ranked_docs[:top_n]]


def boolean_search_sqlite(query: str, conn: sqlite3.Connection, top_n=5):
    """
    Boolean search using inverted_index table in SQLite.
    (conn: sqlite3.Connection to doctor_who.db)
    """
    query_tokens = preprocess_text(query)
    if not query_tokens:
        return []

    doc_scores = Counter()
    cur = conn.cursor()

    for token in query_tokens:
        cur.execute("SELECT doc_id FROM inverted_index WHERE token = ?", (token,))
        rows = cur.fetchall()
        for (doc_id,) in rows:
            doc_scores[doc_id] += 1

    # Sort documents by the number of matching tokens
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    return [doc_id for doc_id, _ in ranked_docs[:top_n]]
