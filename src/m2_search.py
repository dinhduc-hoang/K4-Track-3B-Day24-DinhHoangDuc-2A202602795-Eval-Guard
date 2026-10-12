from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import hashlib
import os, re, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    # Avoid importing the full Transformers stack when the explicitly selected
    # offline fallback is active.  Whitespace tokenisation keeps BM25 usable.
    if os.getenv("RAG_FORCE_FALLBACK") == "1":
        return text.lower()
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text").replace("_", " ")
    except ImportError:
        # Preserve a usable BM25 search when dependencies have not been installed yet.
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = list(chunks)
        self.corpus_tokens = [segment_vietnamese(chunk.get("text", "")).split() for chunk in self.documents]
        if not self.corpus_tokens:
            self.bm25 = None
            return

        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except ImportError:
            self.bm25 = _TokenOverlapScorer(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []

        query_tokens = segment_vietnamese(query).split()
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self.documents[index].get("text", ""),
                score=float(scores[index]),
                metadata=dict(self.documents[index].get("metadata", {})),
                method="bm25",
            )
            for index in top_indices
            if scores[index] > 0
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception as exc:
            print(f"  ⚠️  Qdrant {QDRANT_HOST}:{QDRANT_PORT} unavailable; using in-memory Qdrant: {exc}")
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            if os.getenv("RAG_FORCE_FALLBACK") == "1":
                self._encoder = _HashingEncoder(EMBEDDING_DIM)
            else:
                try:
                    from sentence_transformers import SentenceTransformer
                    self._encoder = SentenceTransformer(EMBEDDING_MODEL)
                except Exception as exc:
                    print(f"  ⚠️  Embedding model unavailable; using hashing fallback: {exc}")
                    self._encoder = _HashingEncoder(EMBEDDING_DIM)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        # Recreating makes an index run idempotent and avoids stale chunks.
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        if not chunks:
            return

        texts = [chunk.get("text", "") for chunk in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=False)
        points = [
            PointStruct(
                id=index,
                vector=vector.tolist(),
                payload={**chunk.get("metadata", {}), "text": chunk.get("text", "")},
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0:
            return []

        query_vector = self._get_encoder().encode(query).tolist()
        if hasattr(self.client, "query_points"):
            points = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k,
            ).points
        else:
            # Day 24 pins qdrant-client<1.10, which predates query_points().
            points = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k,
            )
        return [
            SearchResult(
                text=point.payload.get("text", ""),
                score=float(point.score),
                metadata={key: value for key, value in point.payload.items() if key != "text"},
                method="dense",
            )
            for point in points
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0:
        raise ValueError("k must be non-negative")

    scores: dict[str, tuple[float, SearchResult]] = {}
    for ranked_results in results_list:
        for rank, result in enumerate(ranked_results):
            total, representative = scores.get(result.text, (0.0, result))
            scores[result.text] = (total + 1.0 / (k + rank + 1), representative)

    ranked = sorted(scores.values(), key=lambda item: item[0], reverse=True)
    return [
        SearchResult(text=result.text, score=score, metadata=dict(result.metadata), method="hybrid")
        for score, result in ranked[:top_k]
    ]


class _TokenOverlapScorer:
    """Small offline fallback used only when rank_bm25 is not installed."""

    def __init__(self, corpus_tokens: list[list[str]]):
        self.corpus_tokens = corpus_tokens

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        query_set = set(query_tokens)
        return [float(sum(token in query_set for token in tokens)) for tokens in self.corpus_tokens]


class _HashingEncoder:
    """Deterministic CPU-only embedding fallback for constrained environments."""

    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def encode(self, texts, show_progress_bar: bool = False):
        import numpy as np

        single_text = isinstance(texts, str)
        inputs = [texts] if single_text else texts
        vectors = []
        for text in inputs:
            vector = np.zeros(self.dimensions, dtype=float)
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                vector[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0
            magnitude = np.linalg.norm(vector)
            vectors.append(vector / magnitude if magnitude else vector)
        return vectors[0] if single_text else np.asarray(vectors)


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
