"""Indexing and idempotent re-ingestion.

Order of operations: plan -> embed -> upsert parents -> upsert children -> delete stale.

Per child chunk the :class:`ReingestPlanner` decides:

* **unchanged** - same vector id, same ``embedding_hash`` and ``record_hash``: nothing to do;
* **refreshed** - same id and embedding text, metadata changed (``record_hash``): stored vector reused,
  metadata rewritten;
* **reused** - an existing record (any id of this document) has the same ``embedding_hash``: its
  vector is copied, no model call;
* **embedded** - new text: embedded with the model.

Existing vectors of the document that are not in the new chunk set are **stale** and deleted, as
are parents no longer referenced. Re-ingesting an unchanged file is therefore a no-op.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

from ai.chunking.errors import IndexingError
from ai.chunking.interfaces import ParentStore, StoredRecord, VectorRecord, VectorStore
from ai.chunking.models import ChildChunk, ParentChunk
from ai.embedding.bge import Embedder, validate_vector

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ReingestPlan:
    unchanged: list[ChildChunk] = field(default_factory=list)
    refreshed: list[ChildChunk] = field(default_factory=list)
    reused: list[tuple[ChildChunk, str]] = field(default_factory=list)  # (chunk, source vector id)
    embed: list[ChildChunk] = field(default_factory=list)
    stale_ids: list[str] = field(default_factory=list)


class ReingestPlanner:
    @staticmethod
    def plan(children: Sequence[ChildChunk], existing: Sequence[StoredRecord]) -> ReingestPlan:
        plan = ReingestPlan()
        by_id = {r.id: r for r in existing}
        by_hash: dict[str, str] = {}
        for record in existing:
            h = record.metadata.get("embedding_hash")
            if isinstance(h, str) and h:
                by_hash.setdefault(h, record.id)
        new_ids = set()
        for child in children:
            vid = child.vector_id
            new_ids.add(vid)
            old = by_id.get(vid)
            if old is not None and old.metadata.get("embedding_hash") == child.embedding_hash:
                if old.metadata.get("record_hash") == child.record_hash:
                    plan.unchanged.append(child)
                else:
                    plan.refreshed.append(child)
            elif child.embedding_hash in by_hash:
                plan.reused.append((child, by_hash[child.embedding_hash]))
            else:
                plan.embed.append(child)
        plan.stale_ids = [r.id for r in existing if r.id not in new_ids]
        return plan


@dataclass(slots=True)
class IndexReport:
    document_uid: str
    collection: str
    total: int = 0
    unchanged: int = 0
    refreshed: int = 0
    reused: int = 0
    embedded: int = 0
    deleted: int = 0
    parents_upserted: int = 0
    parents_deleted: int = 0
    seconds: float = 0.0

    @property
    def is_noop(self) -> bool:
        return not (self.refreshed or self.reused or self.embedded or self.deleted or self.parents_deleted)


class IndexingService:
    """Embeds child chunks and writes them (and their parents) to the stores.

    Args:
        embedder: Embedding model (only children are embedded).
        store: Vector store for children.
        parent_store: Store for parent chunks (optional).
        batch_size: Texts per embedding call.
    """

    def __init__(self, embedder: Embedder, store: VectorStore, parent_store: ParentStore | None = None,
                 batch_size: int = 64) -> None:
        self.embedder = embedder
        self.store = store
        self.parent_store = parent_store
        self.batch_size = batch_size
        self.planner = ReingestPlanner()

    def index(self, children: Sequence[ChildChunk], document_uid: str,
              parents: Sequence[ParentChunk] = ()) -> IndexReport:
        start = time.perf_counter()
        for chunk in children:
            if chunk.document_uid != document_uid:
                raise IndexingError(f"Chunk {chunk.chunk_id} belongs to {chunk.document_uid}, not {document_uid}")
            if chunk.embedding_model != self.embedder.model_name:
                raise IndexingError(f"Chunk {chunk.chunk_id} was prepared for {chunk.embedding_model}, "
                                    f"embedder is {self.embedder.model_name}")
        parent_ids = {p.parent_chunk_id for p in parents}
        if self.parent_store is not None:
            dangling = [c.chunk_id for c in children if c.parent_chunk_id not in parent_ids]
            if dangling:
                raise IndexingError(f"{len(dangling)} chunks reference parents that were not supplied")

        existing = self.store.get_document(document_uid)
        plan = self.planner.plan(children, existing)
        report = IndexReport(document_uid, self.store.collection_name, total=len(children),
                             unchanged=len(plan.unchanged), refreshed=len(plan.refreshed),
                             reused=len(plan.reused), embedded=len(plan.embed))

        # 1. vectors: new texts from the model, the rest copied from the store
        vectors: dict[str, list[float]] = {}
        for i in range(0, len(plan.embed), self.batch_size):
            batch = plan.embed[i:i + self.batch_size]
            try:
                embedded = self.embedder.embed_documents([c.embedding_text for c in batch])
            except Exception as exc:
                raise IndexingError(f"Embedding failed: {exc}") from exc
            if len(embedded) != len(batch):
                raise IndexingError("Embedder returned a different number of vectors")
            for chunk, vector in zip(batch, embedded):
                vectors[chunk.vector_id] = validate_vector(vector, self.embedder.dimension)
        sources = {c.vector_id: c.vector_id for c in plan.refreshed}
        sources.update({c.vector_id: src for c, src in plan.reused})
        if sources:
            stored = {r.id: r.vector for r in self.store.get(sorted(set(sources.values())), include_vectors=True)}
            for vid, src in sources.items():
                vector = stored.get(src)
                if vector is None:
                    raise IndexingError(f"Stored vector {src} disappeared during re-ingestion")
                vectors[vid] = vector

        # 2. parents first, so no stored child ever points at a missing parent
        if self.parent_store is not None:
            self.parent_store.upsert(list(parents))
            report.parents_upserted = len(parents)

        # 3. children
        to_write = [c for c in children if c.vector_id in vectors]
        self.store.upsert([VectorRecord(c.vector_id, vectors[c.vector_id], c.text, c.to_metadata())
                           for c in to_write])

        # 4. stale children and parents
        if plan.stale_ids:
            self.store.delete(plan.stale_ids)
            report.deleted = len(plan.stale_ids)
        if self.parent_store is not None:
            stale_parents = self.parent_store.list_ids(document_uid) - parent_ids
            if stale_parents:
                self.parent_store.delete(document_uid, stale_parents)
                report.parents_deleted = len(stale_parents)

        report.seconds = round(time.perf_counter() - start, 3)
        logger.info("Indexed %s: %s", document_uid, report)
        return report

    def remove_document(self, document_uid: str) -> tuple[int, int]:
        """Delete a document's vectors and parents; returns ``(children, parents)`` removed."""
        children = self.store.delete_document(document_uid)
        parents = self.parent_store.delete_document(document_uid) if self.parent_store is not None else 0
        return children, parents
