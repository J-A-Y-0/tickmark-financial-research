"""Document store with hybrid retrieval: BM25 (SQLite FTS5) + vectors, fused with RRF.

Access control is applied inside retrieval: a chunk whose document the caller's role may
not read is never scored, so it can never reach a prompt, a citation or the cache.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass

import numpy as np

from tickmark.db import Database
from tickmark.models import Evidence, Route
from tickmark.retrieval.chunking import Chunk
from tickmark.retrieval.embeddings import Embedder
from tickmark.text import content_words

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    collection TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL DEFAULT '',
    origin TEXT NOT NULL DEFAULT 'corpus',          -- corpus | upload
    meta TEXT NOT NULL DEFAULT '{}',
    allowed_roles TEXT,                              -- JSON list; NULL = anyone allowed on the route
    status TEXT NOT NULL DEFAULT 'indexed',          -- pending | parsing | embedding | indexed | failed
    progress INTEGER NOT NULL DEFAULT 100,
    error TEXT,
    uploaded_by TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    locator TEXT NOT NULL,
    text TEXT NOT NULL,
    search_text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    search_text, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, search_text) VALUES (new.id, new.search_text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, search_text) VALUES ('delete', old.id, old.search_text);
END;
CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

RRF_K = 60
CANDIDATES = 40
_YEAR = re.compile(r"\b(19[89]\d|20\d\d)\b")


@dataclass
class DocumentInfo:
    doc_id: str
    collection: str
    title: str
    url: str
    origin: str
    meta: dict
    allowed_roles: list[str] | None
    status: str
    progress: int
    error: str | None
    uploaded_by: str | None
    created_at: float

    def readable_by(self, role: str) -> bool:
        return self.allowed_roles is None or role in self.allowed_roles

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "collection": self.collection,
            "title": self.title,
            "url": self.url,
            "origin": self.origin,
            "allowed_roles": self.allowed_roles,
            "status": self.status,
            "progress": self.progress,
            "error": self.error,
            "uploaded_by": self.uploaded_by,
            "created_at": self.created_at,
            "fiscal_year": self.meta.get("fiscal_year"),
            "ticker": self.meta.get("ticker"),
            "doc_type": self.meta.get("doc_type") or self.meta.get("form"),
            "date": self.meta.get("date") or self.meta.get("filing_date"),
        }


def _row_to_doc(row) -> DocumentInfo:
    return DocumentInfo(
        doc_id=row["doc_id"],
        collection=row["collection"],
        title=row["title"],
        url=row["url"],
        origin=row["origin"],
        meta=json.loads(row["meta"]),
        allowed_roles=json.loads(row["allowed_roles"]) if row["allowed_roles"] else None,
        status=row["status"],
        progress=row["progress"],
        error=row["error"],
        uploaded_by=row["uploaded_by"],
        created_at=row["created_at"],
    )


def search_header(meta: dict, title: str, locator: str) -> str:
    """Context prepended to a chunk for indexing, so a table row still matches 'Uber 2021'."""
    parts = [title]
    if meta.get("ticker"):
        parts.append(meta["ticker"])
    if meta.get("fiscal_year"):
        parts.append(f"fiscal year {meta['fiscal_year']}")
    parts.append(locator)
    return " | ".join(parts)


class DocumentStore:
    def __init__(self, db: Database, embedder: Embedder):
        self.db = db
        self.embedder = embedder
        with db.tx() as conn:
            conn.executescript(SCHEMA)
        self._vectors: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._lock = threading.Lock()

    # ---- metadata ---------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        rows = self.db.query("SELECT value FROM index_meta WHERE key = ?", (key,))
        return rows[0]["value"] if rows else None

    def set_meta(self, key: str, value: str) -> None:
        with self.db.tx() as conn:
            conn.execute("INSERT OR REPLACE INTO index_meta(key, value) VALUES (?, ?)", (key, value))

    # ---- documents --------------------------------------------------------------

    def create_document(
        self,
        doc_id: str,
        collection: str,
        title: str,
        *,
        url: str = "",
        origin: str = "corpus",
        meta: dict | None = None,
        allowed_roles: list[str] | None = None,
        status: str = "pending",
        uploaded_by: str | None = None,
    ) -> None:
        with self.db.tx() as conn:
            conn.execute(
                """INSERT INTO documents(doc_id, collection, title, url, origin, meta, allowed_roles,
                                         status, progress, uploaded_by, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)""",
                (
                    doc_id,
                    collection,
                    title,
                    url,
                    origin,
                    json.dumps(meta or {}),
                    json.dumps(allowed_roles) if allowed_roles is not None else None,
                    status,
                    uploaded_by,
                    time.time(),
                ),
            )

    def set_status(self, doc_id: str, status: str, progress: int, error: str | None = None) -> None:
        with self.db.tx() as conn:
            conn.execute(
                "UPDATE documents SET status = ?, progress = ?, error = ? WHERE doc_id = ?",
                (status, progress, error, doc_id),
            )
        doc = self.get_document(doc_id)
        if doc:
            self._invalidate(doc.collection)

    def add_chunks(self, doc_id: str, chunks: list[Chunk]) -> list[int]:
        """Embed and store chunks. Returns the new chunk ids."""
        doc = self.get_document(doc_id)
        if doc is None:
            raise KeyError(doc_id)
        search_texts = [f"{search_header(doc.meta, doc.title, c.locator)}\n{c.text}" for c in chunks]
        vectors = self.embedder.embed(search_texts) if chunks else np.zeros((0, 1), np.float32)
        ids: list[int] = []
        with self.db.tx() as conn:
            for chunk, search_text, vector in zip(chunks, search_texts, vectors, strict=True):
                cursor = conn.execute(
                    """INSERT INTO chunks(doc_id, ordinal, locator, text, search_text, embedding)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (doc_id, chunk.ordinal, chunk.locator, chunk.text, search_text, vector.tobytes()),
                )
                ids.append(cursor.lastrowid)
        self._invalidate(doc.collection)
        return ids

    def delete_document(self, doc_id: str) -> None:
        doc = self.get_document(doc_id)
        with self.db.tx() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        if doc:
            self._invalidate(doc.collection)

    def delete_origin(self, origin: str) -> None:
        for row in self.db.query("SELECT doc_id FROM documents WHERE origin = ?", (origin,)):
            self.delete_document(row["doc_id"])

    def get_document(self, doc_id: str) -> DocumentInfo | None:
        rows = self.db.query("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        return _row_to_doc(rows[0]) if rows else None

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        if collection:
            rows = self.db.query("SELECT * FROM documents WHERE collection = ? ORDER BY title", (collection,))
        else:
            rows = self.db.query("SELECT * FROM documents ORDER BY collection, title")
        return [_row_to_doc(r) for r in rows]

    def document_chunks(self, doc_id: str) -> list[dict]:
        rows = self.db.query("SELECT ordinal, locator, text FROM chunks WHERE doc_id = ? ORDER BY ordinal", (doc_id,))
        return [dict(r) for r in rows]

    def index_version(self) -> str:
        """Changes whenever the searchable content changes (corpus rebuild, upload, deletion)."""
        row = self.db.query(
            "SELECT COUNT(*) AS n, COALESCE(MAX(created_at), 0) AS latest FROM documents WHERE status = 'indexed'"
        )[0]
        fingerprint = (self.get_meta("corpus_fingerprint") or "")[:12]
        return f"{fingerprint}-{row['n']}-{row['latest']:.0f}"

    def chunk_count(self) -> int:
        return self.db.query("SELECT COUNT(*) AS n FROM chunks")[0]["n"]

    def probe(self, chunk_id: int) -> bool:
        """Read-your-write check: the chunk is findable by full-text search and its vector is intact."""
        rows = self.db.query("SELECT search_text, embedding FROM chunks WHERE id = ?", (chunk_id,))
        if not rows:
            return False
        terms = _fts_query(rows[0]["search_text"])
        in_fts = bool(
            terms and self.db.query("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? AND rowid = ?", (terms, chunk_id))
        )
        stored = np.frombuffer(rows[0]["embedding"], dtype=np.float32)
        expected = self.embedder.embed([rows[0]["search_text"]])[0]
        return in_fts and stored.shape == expected.shape and float(stored @ expected) > 0.99

    # ---- retrieval --------------------------------------------------------------

    def _invalidate(self, collection: str) -> None:
        with self._lock:
            self._vectors.pop(collection, None)

    def _collection_vectors(self, collection: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            if collection not in self._vectors:
                rows = self.db.query(
                    """SELECT c.id, c.doc_id, c.embedding FROM chunks c JOIN documents d USING(doc_id)
                       WHERE d.collection = ? AND d.status = 'indexed' ORDER BY c.id""",
                    (collection,),
                )
                ids = np.array([r["id"] for r in rows], dtype=np.int64)
                docs = np.array([r["doc_id"] for r in rows], dtype=object)
                matrix = (
                    np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
                    if rows
                    else np.zeros((0, 1), dtype=np.float32)
                )
                self._vectors[collection] = (ids, docs, matrix)
            return self._vectors[collection]

    def scope_documents(self, query: str, collection: str, role: str) -> list[DocumentInfo]:
        """Documents the role may read, narrowed by companies and fiscal years named in the query.

        The narrowing is deterministic metadata filtering: "Uber revenue in 2021" searches
        only Uber's FY2021 10-K. A filter that would match nothing is dropped rather than
        returning no results (FY2023 figures live in the FY2024 filing's comparatives).
        """
        docs = [d for d in self.list_documents(collection) if d.status == "indexed" and d.readable_by(role)]
        companies = named_companies(query, docs)
        if companies:
            docs = [d for d in docs if d.meta.get("ticker") in companies]
        years = {int(y) for y in _YEAR.findall(query)}
        if years:
            dated = [d for d in docs if d.meta.get("fiscal_year") in years]
            if dated:
                docs = dated
        return docs

    def search(self, query: str, collection: str, role: str, k: int = 6) -> list[Evidence]:
        scoped = self.scope_documents(query, collection, role)
        companies = sorted(named_companies(query, scoped))
        if len(companies) > 1:
            # A comparison ("Uber vs Lyft"): search each company separately and interleave,
            # so the company with more matching text cannot crowd out the other.
            per = max(2, -(-k // len(companies)))
            groups = [
                self._search_docs(query, collection, [d for d in scoped if d.meta.get("ticker") == t], per) for t in companies
            ]
            merged = [ev for batch in zip(*[g + [None] * (per - len(g)) for g in groups], strict=True) for ev in batch if ev]
            return merged[:k]
        return self._search_docs(query, collection, scoped, k)

    def _search_docs(self, query: str, collection: str, scoped: list[DocumentInfo], k: int) -> list[Evidence]:
        docs = {d.doc_id: d for d in scoped}
        if not docs:
            return []
        ranked: dict[int, float] = {}

        terms = _fts_query(query)
        if terms:
            placeholders = ",".join("?" * len(docs))
            rows = self.db.query(
                f"""SELECT c.id FROM chunks_fts f JOIN chunks c ON c.id = f.rowid
                    WHERE chunks_fts MATCH ? AND c.doc_id IN ({placeholders})
                    ORDER BY bm25(chunks_fts) LIMIT {CANDIDATES}""",
                (terms, *docs),
            )
            for rank, row in enumerate(rows):
                ranked[row["id"]] = ranked.get(row["id"], 0.0) + 1.0 / (RRF_K + rank + 1)

        ids, doc_ids, matrix = self._collection_vectors(collection)
        if len(ids):
            mask = np.isin(doc_ids, list(docs))
            if mask.any():
                query_vector = self.embedder.embed([query])[0]
                scores = matrix[mask] @ query_vector
                allowed_ids = ids[mask]
                for rank, index in enumerate(np.argsort(-scores)[:CANDIDATES]):
                    chunk_id = int(allowed_ids[index])
                    ranked[chunk_id] = ranked.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        top = sorted(ranked.items(), key=lambda item: -item[1])[:k]
        if not top:
            return []
        placeholders = ",".join("?" * len(top))
        rows = {
            r["id"]: r
            for r in self.db.query(
                f"SELECT id, doc_id, ordinal, locator, text FROM chunks WHERE id IN ({placeholders})",
                [chunk_id for chunk_id, _ in top],
            )
        }
        results = []
        for chunk_id, score in top:
            row = rows[chunk_id]
            doc = docs[row["doc_id"]]
            results.append(
                Evidence(
                    kind=Route(collection),
                    title=doc.title,
                    text=row["text"],
                    url=f"/documents/{doc.doc_id}#chunk-{row['ordinal']}",
                    locator=row["locator"],
                    doc_id=doc.doc_id,
                    score=round(score, 5),
                )
            )
        return results


def named_companies(query: str, docs: list[DocumentInfo]) -> set[str]:
    """Tickers of the companies a query names, by ticker or by the first word of the company name."""
    lowered = query.lower()
    return {
        d.meta["ticker"]
        for d in docs
        if d.meta.get("ticker") and re.search(rf"\b({'|'.join(map(re.escape, _aliases(d.meta)))})\b", lowered)
    }


def _aliases(meta: dict) -> list[str]:
    """Names a query may use for a company: its ticker and the first word of its name."""
    names = [meta["ticker"].lower()]
    if meta.get("company"):
        names.append(meta["company"].split()[0].lower().rstrip(","))
    return names


SYNONYMS = {
    "grow": ["growth", "grew", "increase"],
    "growth": ["grew", "increase"],
    "profit": ["income", "profitable"],
    "profitable": ["income", "profit"],
    "sales": ["revenue"],
    "earnings": ["income"],
    "staff": ["employees"],
    "workforce": ["employees"],
}


def _fts_query(text: str) -> str:
    """An FTS5 OR-query of the distinct content words (plus a few financial synonyms), quoted."""
    seen: list[str] = []
    for word in content_words(text):
        for token in re.split(r"[^a-z0-9]+", word):
            for term in [token, *SYNONYMS.get(token, [])]:
                if term and term not in seen:
                    seen.append(term)
    return " OR ".join(f'"{t}"' for t in seen[:32])
