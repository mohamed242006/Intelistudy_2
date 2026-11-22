"""Storage contracts used by indexing (and later retrieval), plus shared scope/filter validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

from ai.chunking.errors import VectorStoreError
from ai.chunking.models import ParentChunk

Scalar = str | int | float | bool
SCOPE_KEYS = frozenset({"user_id", "course_id", "document_uid"})
RESERVED_KEYS = SCOPE_KEYS | {"chunk_id", "embedding_hash", "record_hash"}


@dataclass(slots=True)
class VectorRecord:
    id: str
    vector: list[float]
    document: str
    metadata: dict[str, Scalar]


@dataclass(slots=True)
class StoredRecord:
    id: str
    metadata: dict[str, Any]
    document: str = ""
    vector: list[float] | None = None


@dataclass(slots=True)
class QueryHit:
    id: str
    score: float  # cosine similarity, higher is better
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    collection_name: str

    def upsert(self, records: Sequence[VectorRecord]) -> None: ...

    def get(self, ids: Sequence[str], include_vectors: bool = False) -> list[StoredRecord]: ...

    def get_document(self, document_uid: str, include_vectors: bool = False) -> list[StoredRecord]: ...

    def delete(self, ids: Sequence[str]) -> None: ...

    def delete_document(self, document_uid: str) -> int: ...

    def query(self, vector: Sequence[float], user_id: str, course_id: str, k: int = 5,
              document_uids: Sequence[str] | None = None, filters: dict[str, Scalar] | None = None
              ) -> list[QueryHit]: ...

    def count(self) -> int: ...


@runtime_checkable
class ParentStore(Protocol):
    def upsert(self, parents: Sequence[ParentChunk]) -> None: ...

    def get(self, document_uid: str, parent_ids: Iterable[str]) -> dict[str, ParentChunk]: ...

    def list_ids(self, document_uid: str) -> set[str]: ...

    def delete(self, document_uid: str, parent_ids: Iterable[str]) -> None: ...

    def delete_document(self, document_uid: str) -> int: ...


def _check_scalar(key: str, value: Any) -> None:
    if not isinstance(value, (str, int, float, bool)) or isinstance(value, dict):
        raise VectorStoreError(f"Filter value for '{key}' must be a scalar, got {type(value).__name__}")
    if isinstance(value, str) and value.startswith("$"):
        raise VectorStoreError(f"Filter value for '{key}' must not start with '$'")


def scope_conditions(user_id: str, course_id: str, document_uids: Sequence[str] | None = None,
                     filters: dict[str, Scalar] | None = None) -> list[tuple[str, str, Any]]:
    """Validated ``(key, op, value)`` conditions; ``op`` is ``"eq"`` or ``"in"``.

    ``user_id`` and ``course_id`` are mandatory; extra filters may not use operators or override
    the scope keys.
    """
    for name, value in (("user_id", user_id), ("course_id", course_id)):
        if not isinstance(value, str) or not value.strip():
            raise VectorStoreError(f"{name} is required for every query")
        _check_scalar(name, value)
    conditions: list[tuple[str, str, Any]] = [("user_id", "eq", user_id), ("course_id", "eq", course_id)]
    if document_uids is not None:
        uids = list(document_uids)
        if not uids:
            raise VectorStoreError("document_uids, when given, must not be empty")
        for uid in uids:
            _check_scalar("document_uid", uid)
        conditions.append(("document_uid", "in", uids) if len(uids) > 1 else ("document_uid", "eq", uids[0]))
    for key, value in (filters or {}).items():
        if not isinstance(key, str) or not key or key.startswith("$"):
            raise VectorStoreError(f"Invalid filter key {key!r}")
        if key in RESERVED_KEYS:
            raise VectorStoreError(f"Filter key '{key}' is reserved")
        _check_scalar(key, value)
        conditions.append((key, "eq", value))
    return conditions


def matches(metadata: dict[str, Any], conditions: list[tuple[str, str, Any]]) -> bool:
    for key, op, value in conditions:
        actual = metadata.get(key)
        if op == "eq" and actual != value:
            return False
        if op == "in" and actual not in value:
            return False
    return True
