"""Hybrid retrieval combining dense (vector) and sparse (BM25) scores."""

import os
from typing import List, Dict
from dataclasses import dataclass
import numpy as np

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text

from ..storage.models import ChunkTable
from ..ingestion.bm25_index import BM25Index, BM25Result


@dataclass
class RetrievalConfig:
    """Configuration for hybrid retrieval."""
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    top_k: int = 20
    rerank_top_k: int = 5

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        """Load config from environment."""
        return cls(
            dense_weight=float(os.getenv("DENSE_WEIGHT", "0.5")),
            sparse_weight=float(os.getenv("SPARSE_WEIGHT", "0.5")),
            top_k=int(os.getenv("TOP_K", "20")),
            rerank_top_k=int(os.getenv("RERANK_TOP_K", "5")),
        )


@dataclass
class RetrievedChunk:
    """A retrieved chunk with hybrid score."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DenseRetriever:
    """Vector-based dense retrieval using pgvector."""

    def __init__(self, session: AsyncSession, dimension: int = 1536):
        self.session = session
        self.dimension = dimension

    async def retrieve(
        self,
        query_embedding: List[float],
        limit: int = 20,
    ) -> Dict[str, float]:
        """Retrieve chunks by vector similarity.

        Returns dict of chunk_id -> similarity score (0-1, higher is better).
        """
        # Use cosine distance via <=> operator (pgvector)
        # Distance is 0-2, so similarity = 1 - distance/2
        stmt = (
            select(
                ChunkTable.id,
                (1 - ChunkTable.embedding.cosine_distance(query_embedding)).label("score")
            )
            .where(ChunkTable.embedding.isnot(None))
            .order_by(ChunkTable.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )

        result = await self.session.execute(stmt)
        scores = {row.id: float(row.score) for row in result}
        return scores


class SparseRetriever:
    """BM25-based sparse retrieval."""

    def __init__(self, bm25_index: BM25Index):
        self.index = bm25_index

    def retrieve(self, query: str, limit: int = 20) -> Dict[str, float]:
        """Retrieve chunks by BM25 scoring.

        Returns dict of chunk_id -> BM25 score.
        """
        results = self.index.search(query, limit=limit)
        return {r.chunk_id: r.score for r in results}


class HybridRetriever:
    """Hybrid retriever that fuses dense and sparse scores."""

    def __init__(
        self,
        session: AsyncSession,
        bm25_index: BM25Index,
        config: RetrievalConfig | None = None,
    ):
        self.session = session
        self.dense = DenseRetriever(session)
        self.sparse = SparseRetriever(bm25_index)
        self.config = config or RetrievalConfig()

    async def retrieve(
        self,
        query: str,
        query_embedding: List[float],
    ) -> List[RetrievedChunk]:
        """Retrieve and fuse results from dense and sparse retrievers.

        Uses score normalization and weighted fusion.
        """
        # Get scores from both retrievers
        dense_scores = await self.dense.retrieve(query_embedding, self.config.top_k)
        sparse_scores = self.sparse.retrieve(query, self.config.top_k)

        # Normalize scores to 0-1
        dense_norm = self._normalize_scores(dense_scores)
        sparse_norm = self._normalize_scores(sparse_scores)

        # Fuse with weighted combination
        fused = self._fuse_scores(dense_norm, sparse_norm)

        # Sort by combined score
        sorted_results = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        # Fetch full chunk details
        chunk_ids = [cid for cid, _ in sorted_results[:self.config.rerank_top_k]]
        chunks = await self._fetch_chunks(chunk_ids)

        # Attach scores
        result_map = {c.chunk_id: c for c in chunks}
        results = []
        for chunk_id, combined_score in sorted_results:
            if chunk_id in result_map:
                chunk = result_map[chunk_id]
                chunk.score = combined_score
                chunk.dense_score = dense_norm.get(chunk_id)
                chunk.sparse_score = sparse_norm.get(chunk_id)
                results.append(chunk)

        return results[:self.config.rerank_top_k]

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Min-max normalize scores to 0-1 range."""
        if not scores:
            return {}

        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            return {k: 1.0 for k in scores}

        return {
            k: (v - min_val) / (max_val - min_val)
            for k, v in scores.items()
        }

    def _fuse_scores(
        self,
        dense: Dict[str, float],
        sparse: Dict[str, float],
    ) -> Dict[str, float]:
        """Weighted fusion of dense and sparse scores."""
        all_ids = set(dense.keys()) | set(sparse.keys())

        fused = {}
        for chunk_id in all_ids:
            dense_score = dense.get(chunk_id, 0.0)
            sparse_score = sparse.get(chunk_id, 0.0)

            fused[chunk_id] = (
                self.config.dense_weight * dense_score +
                self.config.sparse_weight * sparse_score
            )

        return fused

    async def _fetch_chunks(self, chunk_ids: List[str]) -> List[RetrievedChunk]:
        """Fetch full chunk details from database."""
        if not chunk_ids:
            return []

        stmt = select(ChunkTable).where(ChunkTable.id.in_(chunk_ids))
        result = await self.session.execute(stmt)
        chunks = result.scalars().all()

        return [
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                score=0.0,
                metadata=chunk.metadata or {},
            )
            for chunk in chunks
        ]
