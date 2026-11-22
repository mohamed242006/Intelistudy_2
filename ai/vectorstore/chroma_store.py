"""ChromaDB vector store.

* Persistent client; **one collection per embedding model + chunker major version**
  (``chunks__bge-small-en-v1.5__c1``), so a model change never mixes vector spaces.
* Cosine space; only child chunks are stored; ``id = {document_uid}:{chunk_id}``.
* Every query is scoped by mandatory ``user_id`` + ``course_id`` (validated filters, no operator
  injection, reserved keys protected); the query dimension is checked against the collection.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Sequence

from ai.chunking.config import CHUNKER_MAJOR, EMBEDDING_MODEL
from ai.chunking.errors import VectorStoreError
from ai.chunking.interfaces import QueryHit, Scalar, StoredRecord, VectorRecord, scope_conditions
from ai.embedding.bge import short_model_name

logger = logging.getLogger(__name__)

_INVALID = re.compile(r"[^a-zA-Z0-9._-]+")
MAX_NAME = 63
UPSERT_BATCH = 1000


def collection_name_for(model_name: str = EMBEDDING_MODEL, chunker_major: str = CHUNKER_MAJOR,
                        prefix: str = "chunks") -> str:
    """Valid Chroma collection name for a model + chunker major version.

    Invalid characters become ``-``; names that had to change or are too long get a hash suffix
    so two different models can never collide.
    """
    raw = f"{prefix}__{short_model_name(model_name)}__c{chunker_major}"
    name = _INVALID.sub("-", raw).strip("-._")
    if name != raw or len(name) > MAX_NAME or len(name) < 3:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
        name = f"{name[:MAX_NAME - 9].strip('-._')}-{digest}"
    if not name[0].isalnum():
        name = f"c{name}"[:MAX_NAME]
    return name


def _to_where(conditions: list[tuple[str, str, Any]]) -> dict[str, Any]:
    clauses = [{key: {"$eq": value}} if op == "eq" else {key: {"$in": list(value)}} for key, op, value in conditions]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class ChromaVectorStore:
    """Persistent Chroma collection for child chunks.

    Args:
        path: Directory of the persistent database (keep it outside cloud-synced folders).
        embedding_model: Model whose vectors this collection holds (part of the name).
        dimension: Expected vector dimension (checked on upsert and query).
        client: An existing Chroma client (tests may pass an ephemeral one).
        collection_name: Override the derived name.
    """

    def __init__(self, path: str | Path | None = None, embedding_model: str = EMBEDDING_MODEL,
                 dimension: int | None = 384, client: Any = None, collection_name: str | None = None) -> None:
        import chromadb

        if client is None:
            client = chromadb.PersistentClient(path=str(path)) if path is not None else chromadb.EphemeralClient()
        self.client = client
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.collection_name = collection_name or collection_name_for(embedding_model)
        self.collection = client.get_or_create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": embedding_model,
                      "chunker_major": CHUNKER_MAJOR, **({"dimension": dimension} if dimension else {})},
            embedding_function=None,
        )
        stored_model = (self.collection.metadata or {}).get("embedding_model")
        if stored_model and stored_model != embedding_model:
            raise VectorStoreError(f"Collection {self.collection_name} holds {stored_model}, not {embedding_model}")

    # ------------------------------------------------------------------ write
    def upsert(self, records: Sequence[VectorRecord]) -> None:
        for record in records:
            if self.dimension is not None and len(record.vector) != self.dimension:
                raise VectorStoreError(f"Vector dimension {len(record.vector)} != {self.dimension} ({record.id})")
        for start in range(0, len(records), UPSERT_BATCH):
            batch = records[start:start + UPSERT_BATCH]
            self.collection.upsert(
                ids=[r.id for r in batch],
                embeddings=[list(r.vector) for r in batch],
                documents=[r.document for r in batch],
                metadatas=[{k: v for k, v in r.metadata.items() if v is not None} for r in batch],
            )

    def delete(self, ids: Sequence[str]) -> None:
        ids = list(ids)
        for start in range(0, len(ids), UPSERT_BATCH):
            self.collection.delete(ids=ids[start:start + UPSERT_BATCH])

    def delete_document(self, document_uid: str) -> int:
        ids = [r.id for r in self.get_document(document_uid)]
        self.delete(ids)
        return len(ids)

    # ------------------------------------------------------------------- read
    def _records(self, result: dict[str, Any], include_vectors: bool) -> list[StoredRecord]:
        ids = result.get("ids") or []
        metas = result.get("metadatas")
        docs = result.get("documents")
        vecs = result.get("embeddings") if include_vectors else None
        out: list[StoredRecord] = []
        for i, rid in enumerate(ids):
            vector = [float(x) for x in vecs[i]] if vecs is not None and len(vecs) > i else None
            out.append(StoredRecord(rid, dict(metas[i] or {}) if metas else {}, docs[i] if docs else "", vector))
        return out

    def get(self, ids: Sequence[str], include_vectors: bool = False) -> list[StoredRecord]:
        if not ids:
            return []
        include = ["metadatas", "documents"] + (["embeddings"] if include_vectors else [])
        return self._records(self.collection.get(ids=list(ids), include=include), include_vectors)

    def get_document(self, document_uid: str, include_vectors: bool = False) -> list[StoredRecord]:
        include = ["metadatas", "documents"] + (["embeddings"] if include_vectors else [])
        result = self.collection.get(where={"document_uid": {"$eq": document_uid}}, include=include)
        return self._records(result, include_vectors)

    def query(self, vector: Sequence[float], user_id: str, course_id: str, k: int = 5,
              document_uids: Sequence[str] | None = None, filters: dict[str, Scalar] | None = None
              ) -> list[QueryHit]:
        conditions = scope_conditions(user_id, course_id, document_uids, filters)
        if self.dimension is not None and len(vector) != self.dimension:
            raise VectorStoreError(f"Query dimension {len(vector)} != {self.dimension}")
        if k <= 0 or self.collection.count() == 0:
            return []
        result = self.collection.query(query_embeddings=[list(vector)], n_results=k, where=_to_where(conditions),
                                       include=["documents", "metadatas", "distances"])
        hits: list[QueryHit] = []
        for rid, doc, meta, dist in zip(result["ids"][0], result["documents"][0], result["metadatas"][0],
                                        result["distances"][0]):
            hits.append(QueryHit(rid, round(1.0 - float(dist), 6), doc or "", dict(meta or {})))
        return hits

    def count(self) -> int:
        return int(self.collection.count())
