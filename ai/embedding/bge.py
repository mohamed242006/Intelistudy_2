"""Embedders.

:class:`BgeEmbedder` wraps ``BAAI/bge-small-en-v1.5`` (384-d) through sentence-transformers:

* the model is loaded lazily (first call), so importing this module is cheap;
* documents are embedded as-is, queries with the BGE retrieval instruction;
* vectors are L2-normalised (cosine = dot product); NaN / inf / zero vectors are rejected;
* identical texts are embedded once per call.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Protocol, Sequence, runtime_checkable

from ai.chunking.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingError(Exception):
    """The model produced an invalid vector or could not be loaded."""


@runtime_checkable
class Embedder(Protocol):
    """What indexing and retrieval need from an embedding model."""

    model_name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


def validate_vector(vector: Sequence[float], dimension: int | None = None) -> list[float]:
    """Return a normalised copy of ``vector``; raise :class:`EmbeddingError` if it is unusable."""
    values = [float(v) for v in vector]
    if dimension is not None and len(values) != dimension:
        raise EmbeddingError(f"Vector has dimension {len(values)}, expected {dimension}")
    if not all(math.isfinite(v) for v in values):
        raise EmbeddingError("Vector contains NaN or infinity")
    norm = math.sqrt(sum(v * v for v in values))
    if norm < 1e-12:
        raise EmbeddingError("Zero vector")
    return [v / norm for v in values]


class BgeEmbedder:
    """BGE sentence embedder.

    Args:
        model_name: Hugging Face model id.
        device: ``"cpu"``, ``"cuda"``... (``None`` lets sentence-transformers decide).
        batch_size: Encoding batch size.
        query_instruction: Prefix for queries (BGE v1.5 retrieval instruction).
        local_files_only: Do not download the model (use the local Hugging Face cache).
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str | None = None,
        batch_size: int = 32,
        query_instruction: str = BGE_QUERY_INSTRUCTION,
        local_files_only: bool = False,
        dimension: int = 384,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.local_files_only = local_files_only
        self.dimension = dimension
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    model = self._load()
                    get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
                    dim = get_dim()
                    if dim:
                        self.dimension = int(dim)
                    self._model = model
                    logger.info("Loaded %s (%d-d) on %s", self.model_name, self.dimension, model.device)
        return self._model

    def _load(self) -> Any:
        """Load from the local Hugging Face cache first; download only if allowed and needed."""
        from sentence_transformers import SentenceTransformer

        attempts = (True,) if self.local_files_only else (True, False)
        error: Exception | None = None
        for local_only in attempts:
            try:
                return SentenceTransformer(self.model_name, device=self.device, local_files_only=local_only)
            except Exception as exc:  # not cached / offline / corrupt
                error = exc
        raise EmbeddingError(f"Cannot load embedding model {self.model_name}: {error}") from error

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        unique = list(dict.fromkeys(texts))
        raw = self.model.encode(unique, batch_size=self.batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
        vectors = {text: validate_vector(vec.tolist(), self.dimension) for text, vec in zip(unique, raw)}
        return [vectors[t] for t in texts]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages (no instruction)."""
        return self._encode([t if t.strip() else " " for t in texts])

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (with the BGE retrieval instruction)."""
        if not text.strip():
            raise EmbeddingError("Empty query")
        return self._encode([f"{self.query_instruction}{text}"])[0]


def short_model_name(model_name: str) -> str:
    """``BAAI/bge-small-en-v1.5`` -> ``bge-small-en-v1.5`` (used in collection names)."""
    return model_name.rstrip("/").split("/")[-1]
