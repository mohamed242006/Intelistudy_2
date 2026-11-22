"""Embedding validation, vector/parent stores, scoping, re-ingestion."""

from __future__ import annotations

import math
import os
import unittest

from ai.chunking.engine import ChunkingEngine
from ai.chunking.errors import IndexingError, VectorStoreError
from ai.chunking.indexing import IndexingService
from ai.chunking.interfaces import VectorRecord
from ai.chunking.memory_store import InMemoryParentStore, InMemoryVectorStore
from ai.chunking.models import ChunkContext
from ai.chunking.tokens import default_token_counter
from ai.embedding.bge import EmbeddingError, validate_vector
from ai.ingestion.ingestion_pipeline import IngestionPipeline
from ai.pipeline import DocumentIndexer, make_document_uid
from ai.storage.sqlite_parent_store import SQLiteParentStore
from ai.vectorstore.chroma_store import ChromaVectorStore, collection_name_for
from tests.fixtures import FakeEmbedder, FakeOCR, make_docx, make_pdf, tmpdir

COUNTER = default_token_counter()


def chunk(path, user="u1", course="c1", uid=None):  # noqa: ANN001, ANN201
    doc = IngestionPipeline(ocr=FakeOCR()).ingest(path)
    uid = uid or make_document_uid(user, course, doc.source.doc_id)
    return ChunkingEngine(tokenizer=COUNTER).run(doc, ChunkContext(user, course, uid, "BAAI/bge-small-en-v1.5")), uid


class VectorValidationTests(unittest.TestCase):
    def test_validate_vector(self) -> None:
        v = validate_vector([3.0, 4.0])
        self.assertAlmostEqual(math.hypot(*v), 1.0)
        for bad in ([0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0]):
            with self.assertRaises(EmbeddingError):
                validate_vector(bad)
        with self.assertRaises(EmbeddingError):
            validate_vector([1.0, 0.0], dimension=3)

    def test_collection_names(self) -> None:
        self.assertEqual(collection_name_for("BAAI/bge-small-en-v1.5", "1"), "chunks__bge-small-en-v1.5__c1")
        odd = collection_name_for("org/model with spaces & symbols" + "x" * 80, "1")
        self.assertLessEqual(len(odd), 63)
        self.assertRegex(odd, r"^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$")
        self.assertNotEqual(collection_name_for("a/b c", "1"), collection_name_for("a/b_c", "1"))


class StoreContractMixin:
    """Shared tests for every VectorStore implementation."""

    def make_store(self):  # noqa: ANN201
        raise NotImplementedError

    def test_scope_is_mandatory_and_filters_validated(self) -> None:
        store = self.make_store()
        store.upsert([VectorRecord("d1:c1", [1.0] + [0.0] * 63, "text",
                                   {"user_id": "u1", "course_id": "c1", "document_uid": "d1", "kind": "text"})])
        vec = [1.0] + [0.0] * 63
        self.assertEqual(len(store.query(vec, "u1", "c1")), 1)
        self.assertEqual(store.query(vec, "u2", "c1"), [])
        self.assertEqual(store.query(vec, "u1", "c2"), [])
        self.assertEqual(len(store.query(vec, "u1", "c1", filters={"kind": "text"})), 1)
        for args in (("", "c1"), ("u1", "")):
            with self.assertRaises(VectorStoreError):
                store.query(vec, *args)
        for bad in ({"user_id": "u2"}, {"$or": "x"}, {"kind": {"$ne": "x"}}, {"kind": "$x"}):
            with self.assertRaises(VectorStoreError):
                store.query(vec, "u1", "c1", filters=bad)
        with self.assertRaises(VectorStoreError):
            store.query([1.0, 0.0], "u1", "c1")

    def test_index_reindex_edit_remove(self) -> None:
        d = tmpdir()
        store = self.make_store()
        parents = SQLiteParentStore(d / "p.sqlite3")
        embedder = FakeEmbedder()
        service = IndexingService(embedder, store, parents)

        result, uid = chunk(make_docx(d / "notes.docx"))
        first = service.index(result.children, uid, result.parents)
        self.assertEqual((first.embedded, first.deleted), (len(result.children), 0))
        self.assertEqual(store.count(), len(result.children))
        self.assertEqual(parents.list_ids(uid), {p.parent_chunk_id for p in result.parents})

        again = service.index(result.children, uid, result.parents)  # unchanged file: no-op
        self.assertTrue(again.is_noop)
        self.assertEqual(again.unchanged, len(result.children))

        calls = embedder.texts_embedded
        edited_path = make_docx(d / "edited.docx", extra_paragraph="A newly added paragraph about Raft terms.")
        self.assertNotEqual(uid, chunk(edited_path)[1])  # different file hash -> different default upload id
        edited, _ = chunk(edited_path, uid=uid)  # the user replaced the file of the same upload
        report = service.index(edited.children, uid, edited.parents)  # re-ingest under the same upload
        self.assertGreater(report.unchanged + report.refreshed + report.reused, 0)
        self.assertLess(embedder.texts_embedded - calls, len(edited.children))
        self.assertEqual({r.id for r in store.get_document(uid)}, {c.vector_id for c in edited.children})
        self.assertEqual(parents.list_ids(uid), {p.parent_chunk_id for p in edited.parents})

        hits = store.query(embedder.embed_query("newly added paragraph Raft terms"), "u1", "c1", k=3)
        self.assertIn("newly added", hits[0].document)
        self.assertIn("citation_label", hits[0].metadata)

        removed = service.remove_document(uid)
        self.assertEqual(removed, (len(edited.children), len(edited.parents)))
        self.assertEqual(store.count(), 0)

    def test_same_file_two_users_no_collision(self) -> None:
        d = tmpdir()
        path = make_pdf(d / "shared.pdf")
        store = self.make_store()
        parents = InMemoryParentStore()
        service = IndexingService(FakeEmbedder(), store, parents)
        for user in ("alice", "bob"):
            result, uid = chunk(path, user=user)
            service.index(result.children, uid, result.parents)
        total = store.count()
        result, uid_alice = chunk(path, user="alice")
        self.assertEqual(total, 2 * len(result.children))
        service.remove_document(uid_alice)
        self.assertEqual(store.count(), len(result.children))
        hits = store.query(FakeEmbedder().embed_query("cache"), "alice", "c1")
        self.assertEqual(hits, [])

    def test_wrong_document_or_model_rejected(self) -> None:
        d = tmpdir()
        result, uid = chunk(make_pdf(d / "a.pdf"))
        service = IndexingService(FakeEmbedder(), self.make_store(), InMemoryParentStore())
        with self.assertRaises(IndexingError):
            service.index(result.children, "other-uid", result.parents)
        with self.assertRaises(IndexingError):
            service.index(result.children, uid, [])  # dangling parents
        with self.assertRaises(IndexingError):
            IndexingService(FakeEmbedder(model_name="other/model"), self.make_store()).index(result.children, uid)


class InMemoryStoreTests(StoreContractMixin, unittest.TestCase):
    def make_store(self):  # noqa: ANN201
        return InMemoryVectorStore(dimension=64)


class ChromaStoreTests(StoreContractMixin, unittest.TestCase):
    def make_store(self):  # noqa: ANN201
        return ChromaVectorStore(path=tmpdir() / "chroma", dimension=64)

    def test_persistence_across_clients(self) -> None:
        path = tmpdir() / "chroma"
        store = ChromaVectorStore(path=path, dimension=64)
        store.upsert([VectorRecord("d:c", [0.0, 1.0] + [0.0] * 62, "persisted",
                                   {"user_id": "u", "course_id": "c", "document_uid": "d"})])
        reopened = ChromaVectorStore(path=path, dimension=64)
        self.assertEqual(reopened.count(), 1)

    def test_model_mismatch_rejected(self) -> None:
        path = tmpdir() / "chroma"
        ChromaVectorStore(path=path, dimension=64, collection_name="shared")
        with self.assertRaises(VectorStoreError):
            ChromaVectorStore(path=path, embedding_model="other/model", dimension=64, collection_name="shared")


@unittest.skipUnless(os.environ.get("INTELLISTUDY_MODEL_TESTS") == "1", "set INTELLISTUDY_MODEL_TESTS=1")
class RealModelTests(unittest.TestCase):
    def test_bge_end_to_end(self) -> None:
        d = tmpdir()
        indexer = DocumentIndexer.persistent(d / "index", ingestion=IngestionPipeline(ocr=FakeOCR()))
        outcome = indexer.index_file(make_docx(d / "notes.docx"), "u1", "c1")
        self.assertEqual(outcome.report.embedded, outcome.report.total)
        hits = indexer.search("How do machines agree on a value?", "u1", "c1", k=1)
        self.assertIn("Consensus", hits[0].metadata["citation_label"])
        self.assertGreater(hits[0].score, 0.5)
        self.assertTrue(indexer.index_file(d / "notes.docx", "u1", "c1").report.is_noop)


if __name__ == "__main__":
    unittest.main()
