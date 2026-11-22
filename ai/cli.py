"""Command line: index documents and run a test search.

    python -m ai.cli index data/uploads/test2.pdf --user u1 --course c1 --db C:/intellistudy_index
    python -m ai.cli search "What is a join point?" --user u1 --course c1 --db C:/intellistudy_index
    python -m ai.cli chunks data/uploads/test2.pdf            # ingest + chunk only, print chunks
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _print(text: str) -> None:
    sys.stdout.buffer.write((text + "\n").encode("utf-8", "replace"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ai.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    p_index = sub.add_parser("index", help="ingest, chunk, embed and store files")
    p_index.add_argument("files", nargs="+")
    p_search = sub.add_parser("search", help="scoped vector search")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    for p in (p_index, p_search):
        p.add_argument("--user", required=True)
        p.add_argument("--course", required=True)
        p.add_argument("--db", required=True, help="index directory (keep it outside cloud-synced folders)")
    p_chunks = sub.add_parser("chunks", help="ingest + chunk one file and print the chunks")
    p_chunks.add_argument("file")
    p_chunks.add_argument("--json", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.command == "chunks":
        from ai.chunking.config import EMBEDDING_MODEL
        from ai.chunking.engine import ChunkingEngine
        from ai.chunking.models import ChunkContext
        from ai.ingestion.ingestion_pipeline import IngestionPipeline

        document = IngestionPipeline().ingest(args.file)
        assert document.source is not None
        result = ChunkingEngine().run(document, ChunkContext("cli", "cli", f"d-{document.source.doc_id}",
                                                             EMBEDDING_MODEL))
        if args.json:
            _print(json.dumps([c.to_dict() for c in result.children], ensure_ascii=False, indent=1))
            return 0
        _print(json.dumps(document.statistics, ensure_ascii=False))
        _print(json.dumps(result.statistics, ensure_ascii=False))
        for c in result.children:
            _print(f"\n=== {c.chunk_id} | {c.citation_label} | {c.kind} | {c.token_count} tokens")
            _print(c.text)
        return 0

    from ai.pipeline import DocumentIndexer

    indexer = DocumentIndexer.persistent(Path(args.db))
    if args.command == "index":
        for file in args.files:
            outcome = indexer.index_file(file, args.user, args.course)
            r = outcome.report
            _print(f"{Path(file).name}: uid={outcome.document_uid} chunks={r.total} embedded={r.embedded} "
                   f"reused={r.reused} refreshed={r.refreshed} unchanged={r.unchanged} deleted={r.deleted} "
                   f"parents={r.parents_upserted} ({r.seconds}s)")
        return 0
    for hit in indexer.search(args.query, args.user, args.course, k=args.k):
        meta = hit.metadata
        _print(f"\n[{hit.score:.3f}] {meta.get('file_name')} | {meta.get('citation_label')} | {meta.get('kind')}")
        _print(hit.document[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
