"""Index the bundled corpus and process user uploads.

Uploads follow an accept-then-process pattern: the request stores the file and returns
immediately with status `pending`; a worker thread parses, chunks, embeds and indexes it,
and only marks it `indexed` after a read-your-write probe finds the new chunks through
both retrieval paths.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tickmark.retrieval.chunking import Chunk, chunk_markdown, chunk_pdf, chunk_text
from tickmark.retrieval.store import DocumentStore
from tickmark.text import parse_front_matter

log = logging.getLogger(__name__)

CORPUS_DIRS = ("filings", "knowledge")
UPLOAD_TYPES = {".pdf", ".md", ".markdown", ".txt"}


def corpus_files(data_dir: Path) -> list[Path]:
    return sorted(p for d in CORPUS_DIRS for p in (data_dir / d).glob("*.md") if p.name.lower() != "readme.md")


def corpus_fingerprint(data_dir: Path, embedder_name: str) -> str:
    digest = hashlib.sha256(embedder_name.encode())
    for path in corpus_files(data_dir):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_corpus_index(store: DocumentStore, data_dir: Path, *, force: bool = False) -> bool:
    """(Re)index the bundled corpus when files or the embedder changed. Returns True if rebuilt.

    Uploaded documents are left untouched.
    """
    fingerprint = corpus_fingerprint(data_dir, store.embedder.name)
    if not force and store.get_meta("corpus_fingerprint") == fingerprint:
        return False
    store.delete_origin("corpus")
    for path in corpus_files(data_dir):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        if "doc_id" not in meta or "collection" not in meta:
            raise ValueError(f"{path}: front matter needs doc_id and collection")
        store.create_document(
            meta["doc_id"],
            meta["collection"],
            meta.get("title", path.stem),
            url=meta.get("source_url", ""),
            origin="corpus",
            meta=meta,
            allowed_roles=meta.get("allowed_roles"),
            status="embedding",
        )
        store.add_chunks(meta["doc_id"], chunk_markdown(body))
        store.set_status(meta["doc_id"], "indexed", 100)
    store.set_meta("corpus_fingerprint", fingerprint)
    log.info("Indexed %d corpus documents, %d chunks", len(corpus_files(data_dir)), store.chunk_count())
    return True


def parse_upload(filename: str, data: bytes) -> list[Chunk]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return chunk_pdf(data)
    text = data.decode("utf-8", errors="replace")
    if suffix in {".md", ".markdown"}:
        _, body = parse_front_matter(text)
        return chunk_markdown(body)
    return chunk_text(text)


class UploadWorker:
    """Processes uploads off the request path, one thread pool per process."""

    def __init__(self, store: DocumentStore, max_workers: int = 2):
        self.store = store
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ingest")
        self._lock = threading.Lock()

    def submit(self, filename: str, data: bytes, *, allowed_roles: list[str], uploaded_by: str) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix not in UPLOAD_TYPES:
            raise ValueError(f"Unsupported file type {suffix or '(none)'}; use PDF, Markdown or text")
        stem = re.sub(r"[^a-z0-9]+", "-", Path(filename).stem.lower()).strip("-")[:40] or "upload"
        doc_id = f"upload-{stem}-{uuid.uuid4().hex[:8]}"
        self.store.create_document(
            doc_id,
            "internal",
            Path(filename).name,
            origin="upload",
            meta={"filename": Path(filename).name, "doc_type": "upload"},
            allowed_roles=allowed_roles,
            status="pending",
            uploaded_by=uploaded_by,
        )
        self._pool.submit(self._process, doc_id, filename, data)
        return doc_id

    def _process(self, doc_id: str, filename: str, data: bytes) -> None:
        try:
            self.store.set_status(doc_id, "parsing", 10)
            chunks = parse_upload(filename, data)
            if not chunks:
                raise ValueError(
                    "No extractable text (a scanned PDF needs OCR first)"
                    if filename.lower().endswith(".pdf")
                    else "The file contains no text"
                )
            self.store.set_status(doc_id, "embedding", 50)
            ids = self.store.add_chunks(doc_id, chunks)
            self.store.set_status(doc_id, "embedding", 90)
            if not self.store.probe(ids[0]):
                raise RuntimeError("Indexed chunks were not retrievable (read-your-write probe failed)")
            self.store.set_status(doc_id, "indexed", 100)
        except Exception as exc:  # report every failure on the document, never crash the worker
            log.exception("Upload %s failed", doc_id)
            self.store.set_status(doc_id, "failed", 100, str(exc)[:300])

    def wait(self) -> None:
        """Block until queued uploads finish (tests and CLI)."""
        self._pool.shutdown(wait=True)
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest")
