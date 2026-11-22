"""SQLite store for parent chunks (PostgreSQL planned).

Parents are keyed ``document_uid:parent_chunk_id`` so two users uploading the same file never
collide.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Iterable, Sequence

from ai.chunking.models import ParentChunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS parent_chunks (
    key TEXT PRIMARY KEY,
    document_uid TEXT NOT NULL,
    parent_chunk_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    course_id TEXT NOT NULL,
    record_hash TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_parent_chunks_document ON parent_chunks(document_uid);
"""


class SQLiteParentStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def upsert(self, parents: Sequence[ParentChunk]) -> None:
        rows = [(p.store_key, p.document_uid, p.parent_chunk_id, p.user_id, p.course_id, p.record_hash,
                 json.dumps(p.to_dict(), ensure_ascii=False)) for p in parents]
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT INTO parent_chunks VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET "
                "record_hash = excluded.record_hash, payload = excluded.payload, user_id = excluded.user_id, "
                "course_id = excluded.course_id",
                rows,
            )

    def get(self, document_uid: str, parent_ids: Iterable[str]) -> dict[str, ParentChunk]:
        keys = [f"{document_uid}:{pid}" for pid in parent_ids]
        out: dict[str, ParentChunk] = {}
        with self._lock:
            for start in range(0, len(keys), 500):
                batch = keys[start:start + 500]
                marks = ",".join("?" * len(batch))
                for (payload,) in self._conn.execute(
                        f"SELECT payload FROM parent_chunks WHERE key IN ({marks})", batch):
                    parent = ParentChunk.from_dict(json.loads(payload))
                    out[parent.parent_chunk_id] = parent
        return out

    def list_ids(self, document_uid: str) -> set[str]:
        with self._lock:
            rows = self._conn.execute("SELECT parent_chunk_id FROM parent_chunks WHERE document_uid = ?",
                                      (document_uid,)).fetchall()
        return {r[0] for r in rows}

    def delete(self, document_uid: str, parent_ids: Iterable[str]) -> None:
        keys = [(f"{document_uid}:{pid}",) for pid in parent_ids]
        with self._lock, self._conn:
            self._conn.executemany("DELETE FROM parent_chunks WHERE key = ?", keys)

    def delete_document(self, document_uid: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM parent_chunks WHERE document_uid = ?", (document_uid,))
        return int(cur.rowcount)
