"""In-memory vector and parent stores (tests, notebooks)."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from ai.chunking.errors import VectorStoreError
from ai.chunking.interfaces import QueryHit, Scalar, StoredRecord, VectorRecord, matches, scope_conditions
from ai.chunking.models import ParentChunk


class InMemoryVectorStore:
    def __init__(self, collection_name: str = "memory", dimension: int | None = None) -> None:
        self.collection_name = collection_name
        self.dimension = dimension
        self._records: dict[str, VectorRecord] = {}

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        for record in records:
            if self.dimension is None:
                self.dimension = len(record.vector)
            if len(record.vector) != self.dimension:
                raise VectorStoreError(f"Vector dimension {len(record.vector)} != {self.dimension}")
            self._records[record.id] = VectorRecord(record.id, list(record.vector), record.document,
                                                    dict(record.metadata))

    def _stored(self, record: VectorRecord, include_vectors: bool) -> StoredRecord:
        return StoredRecord(record.id, dict(record.metadata), record.document,
                            list(record.vector) if include_vectors else None)

    def get(self, ids: Sequence[str], include_vectors: bool = False) -> list[StoredRecord]:
        return [self._stored(self._records[i], include_vectors) for i in ids if i in self._records]

    def get_document(self, document_uid: str, include_vectors: bool = False) -> list[StoredRecord]:
        return [self._stored(r, include_vectors) for r in self._records.values()
                if r.metadata.get("document_uid") == document_uid]

    def delete(self, ids: Sequence[str]) -> None:
        for i in ids:
            self._records.pop(i, None)

    def delete_document(self, document_uid: str) -> int:
        ids = [r.id for r in self._records.values() if r.metadata.get("document_uid") == document_uid]
        self.delete(ids)
        return len(ids)

    def query(self, vector: Sequence[float], user_id: str, course_id: str, k: int = 5,
              document_uids: Sequence[str] | None = None, filters: dict[str, Scalar] | None = None
              ) -> list[QueryHit]:
        conditions = scope_conditions(user_id, course_id, document_uids, filters)
        if self.dimension is not None and len(vector) != self.dimension:
            raise VectorStoreError(f"Query dimension {len(vector)} != {self.dimension}")
        qn = math.sqrt(sum(v * v for v in vector)) or 1.0
        hits: list[QueryHit] = []
        for record in self._records.values():
            if not matches(record.metadata, conditions):
                continue
            rn = math.sqrt(sum(v * v for v in record.vector)) or 1.0
            score = sum(a * b for a, b in zip(vector, record.vector)) / (qn * rn)
            hits.append(QueryHit(record.id, score, record.document, dict(record.metadata)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def count(self) -> int:
        return len(self._records)


class InMemoryParentStore:
    def __init__(self) -> None:
        self._parents: dict[str, dict[str, Any]] = {}

    def upsert(self, parents: Sequence[ParentChunk]) -> None:
        for parent in parents:
            self._parents[parent.store_key] = parent.to_dict()

    def get(self, document_uid: str, parent_ids: Iterable[str]) -> dict[str, ParentChunk]:
        out: dict[str, ParentChunk] = {}
        for pid in parent_ids:
            data = self._parents.get(f"{document_uid}:{pid}")
            if data is not None:
                out[pid] = ParentChunk.from_dict(data)
        return out

    def list_ids(self, document_uid: str) -> set[str]:
        return {d["parent_chunk_id"] for d in self._parents.values() if d["document_uid"] == document_uid}

    def delete(self, document_uid: str, parent_ids: Iterable[str]) -> None:
        for pid in parent_ids:
            self._parents.pop(f"{document_uid}:{pid}", None)

    def delete_document(self, document_uid: str) -> int:
        ids = self.list_ids(document_uid)
        self.delete(document_uid, ids)
        return len(ids)
