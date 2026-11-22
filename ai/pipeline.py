"""End-to-end document indexing: ingest -> chunk -> embed -> store.

Usage::

    indexer = DocumentIndexer.persistent("C:/intellistudy_index")   # keep outside OneDrive
    outcome = indexer.index_file("lecture.pdf", user_id="u1", course_id="cse447")
    hits = indexer.search("What is a join point?", user_id="u1", course_id="cse447")
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai.chunking.config import EMBEDDING_MODEL, ChunkingConfig
from ai.chunking.engine import ChunkingEngine
from ai.chunking.indexing import IndexingService, IndexReport
from ai.chunking.interfaces import ParentStore, QueryHit, VectorStore
from ai.chunking.models import ChunkContext, ChunkingResult
from ai.embedding.bge import BgeEmbedder, Embedder
from ai.ingestion.ingestion_pipeline import IngestionPipeline
from ai.ingestion.models import Document

logger = logging.getLogger(__name__)


def make_document_uid(user_id: str, course_id: str, doc_id: str) -> str:
    """Stable id of one document for one user and course (``d-`` + 20 hex chars)."""
    digest = hashlib.sha256(f"{user_id}\x1f{course_id}\x1f{doc_id}".encode("utf-8")).hexdigest()
    return f"d-{digest[:20]}"


@dataclass(slots=True)
class IndexOutcome:
    document: Document
    chunks: ChunkingResult
    report: IndexReport
    document_uid: str


class DocumentIndexer:
    def __init__(self, store: VectorStore, parent_store: ParentStore | None = None,
                 embedder: Embedder | None = None, ingestion: IngestionPipeline | None = None,
                 chunker: ChunkingEngine | None = None) -> None:
        self.embedder = embedder or BgeEmbedder()
        self.ingestion = ingestion or IngestionPipeline()
        self.chunker = chunker or ChunkingEngine(ChunkingConfig())
        self.store = store
        self.parent_store = parent_store
        self.indexing = IndexingService(self.embedder, store, parent_store)

    @classmethod
    def persistent(cls, directory: str | Path, **kwargs: Any) -> "DocumentIndexer":
        """Chroma + SQLite under ``directory``."""
        from ai.storage.sqlite_parent_store import SQLiteParentStore
        from ai.vectorstore.chroma_store import ChromaVectorStore

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        embedder = kwargs.pop("embedder", None) or BgeEmbedder()
        store = ChromaVectorStore(path=directory / "chroma", embedding_model=embedder.model_name,
                                  dimension=embedder.dimension)
        parents = SQLiteParentStore(directory / "parents.sqlite3")
        return cls(store, parents, embedder=embedder, **kwargs)

    def index_file(self, path: str | Path, user_id: str, course_id: str,
                   document_uid: str | None = None) -> IndexOutcome:
        document = self.ingestion.ingest(path)
        assert document.source is not None
        uid = document_uid or make_document_uid(user_id, course_id, document.source.doc_id)
        context = ChunkContext(user_id, course_id, uid, self.embedder.model_name or EMBEDDING_MODEL)
        chunks = self.chunker.run(document, context)
        report = self.indexing.index(chunks.children, uid, chunks.parents)
        return IndexOutcome(document, chunks, report, uid)

    def search(self, query: str, user_id: str, course_id: str, k: int = 5) -> list[QueryHit]:
        """Plain scoped vector search (the full retrieval service is a later layer)."""
        return self.store.query(self.embedder.embed_query(query), user_id=user_id, course_id=course_id, k=k)

    def remove(self, document_uid: str) -> tuple[int, int]:
        return self.indexing.remove_document(document_uid)
