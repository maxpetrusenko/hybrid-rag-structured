"""Retrieval evaluation framework."""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Set, Literal
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..storage.database import Database, get_db
from ..storage.models import ChunkTable
from ..ingestion.bm25_index import BM25Index
from ..ingestion.embeddings import EmbeddingService
from ..retrieval.hybrid import DenseRetriever, SparseRetriever, HybridRetriever, RetrievalConfig
from .metrics import compute_metrics, EvaluationResult, AggregateMetrics


@dataclass
class Query:
    """A test query with relevant documents."""
    query: str
    relevant_docs: Set[str]
    category: str


@dataclass
class MethodResult:
    """Results for a single retrieval method."""
    method: str
    results: List[EvaluationResult]
    aggregate: AggregateMetrics


class Evaluator:
    """Evaluate retrieval methods against a test set."""

    def __init__(
        self,
        session: AsyncSession,
        bm25_index: BM25Index,
        embedder: EmbeddingService,
    ):
        self.session = session
        self.bm25 = bm25_index
        self.embedder = embedder

    async def evaluate(
        self,
        queries: List[Query],
        methods: List[Literal["dense", "sparse", "hybrid"]] = ["dense", "sparse", "hybrid"],
    ) -> List[MethodResult]:
        """Evaluate all retrieval methods."""
        results = []

        for method in methods:
            method_results = []
            for q in queries:
                result = await self._evaluate_query(q, method)
                method_results.append(result)

            aggregate = self._aggregate(method_results)
            results.append(MethodResult(
                method=method,
                results=method_results,
                aggregate=aggregate,
            ))

        return results

    async def _evaluate_query(
        self,
        query: Query,
        method: str,
    ) -> EvaluationResult:
        """Evaluate a single query with a method."""

        if method == "dense":
            retriever = DenseRetriever(self.session)
            query_emb = await self.embedder.embed_single(query.query)
            scores = await retriever.retrieve(query_emb, limit=10)
            retrieved = list(scores.keys())

        elif method == "sparse":
            retriever = SparseRetriever(self.bm25)
            scores = retriever.retrieve(query.query, limit=10)
            retrieved = list(scores.keys())

        else:  # hybrid
            config = RetrievalConfig(dense_weight=0.5, sparse_weight=0.5, top_k=20, rerank_top_k=10)
            retriever = HybridRetriever(self.session, self.bm25, config)
            query_emb = await self.embedder.embed_single(query.query)
            chunks = await retriever.retrieve(query.query, query_emb)
            retrieved = [c.chunk_id for c in chunks]

        # Map chunk IDs back to document names
        doc_names = await self._chunks_to_docs(retrieved)
        relevant_doc_names = await self._resolve_doc_names(query.relevant_docs)

        metrics = compute_metrics(doc_names, relevant_doc_names)

        return EvaluationResult(
            query=query.query,
            retrieved_docs=doc_names,
            relevant_docs=relevant_doc_names,
            **metrics,
        )

    async def _chunks_to_docs(self, chunk_ids: List[str]) -> List[str]:
        """Map chunk IDs to source document names."""
        if not chunk_ids:
            return []

        stmt = (
            select(ChunkTable)
            .where(ChunkTable.id.in_(chunk_ids))
        )
        result = await self.session.execute(stmt)
        chunks = result.scalars().all()

        # Extract filename from metadata source_uri
        doc_names = []
        for chunk in chunks:
            if chunk.metadata and "source_uri" in chunk.metadata:
                # Extract filename from path
                uri = chunk.metadata["source_uri"]
                doc_name = Path(uri).name
                doc_names.append(doc_name)
            else:
                doc_names.append("unknown")

        return doc_names

    async def _resolve_doc_names(self, doc_patterns: Set[str]) -> Set[str]:
        """Resolve document patterns to actual names."""
        # For now, assume patterns are exact filenames
        return doc_patterns

    def _aggregate(self, results: List[EvaluationResult]) -> AggregateMetrics:
        """Aggregate metrics across all queries."""
        n = len(results)
        return AggregateMetrics(
            num_queries=n,
            recall_at_1=sum(r.recall_at_1 for r in results) / n,
            recall_at_5=sum(r.recall_at_5 for r in results) / n,
            recall_at_10=sum(r.recall_at_10 for r in results) / n,
            mrr=sum(r.mrr for r in results) / n,
            ndcg_at_10=sum(r.ndcg_at_10 for r in results) / n,
        )


class RetrievalEval:
    """High-level evaluation runner."""

    @staticmethod
    def load_queries(path: str | Path) -> List[Query]:
        """Load queries from JSONL file."""
        queries = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                queries.append(Query(
                    query=data["query"],
                    relevant_docs=set(data["relevant_docs"]),
                    category=data.get("category", "general"),
                ))
        return queries

    @staticmethod
    async def run(
        queries_path: str | Path,
        output_path: str | Path | None = None,
    ) -> List[MethodResult]:
        """Run full evaluation pipeline."""
        queries = RetrievalEval.load_queries(queries_path)

        await Database.init_db()

        embedder = EmbeddingService()
        bm25 = BM25Index()
        bm25.open_index()

        async with Database.get_session() as session:
            evaluator = Evaluator(session, bm25, embedder)
            results = await evaluator.evaluate(queries)

        await Database.close()

        # Save results if path provided
        if output_path:
            RetrievalEval.save_results(results, output_path)

        return results

    @staticmethod
    def save_results(results: List[MethodResult], path: str | Path) -> None:
        """Save evaluation results to file."""
        output = []

        for method_result in results:
            agg = method_result.aggregate
            output.append({
                "method": method_result.method,
                "num_queries": agg.num_queries,
                "recall_at_1": agg.recall_at_1,
                "recall_at_5": agg.recall_at_5,
                "recall_at_10": agg.recall_at_10,
                "mrr": agg.mrr,
                "ndcg_at_10": agg.ndcg_at_10,
            })

            # Per-query results
            for r in method_result.results:
                output.append({
                    "method": method_result.method,
                    "query": r.query,
                    "recall_at_1": r.recall_at_1,
                    "recall_at_5": r.recall_at_5,
                    "recall_at_10": r.recall_at_10,
                    "mrr": r.mrr,
                    "ndcg_at_10": r.ndcg_at_10,
                })

        with open(path, "w") as f:
            for item in output:
                f.write(json.dumps(item) + "\n")

    @staticmethod
    def format_results(results: List[MethodResult]) -> str:
        """Format results as markdown table."""
        lines = ["# Retrieval Evaluation Results\n"]
        lines.append("| Method | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |")
        lines.append("|--------|----------|----------|-----------|-----|---------|")

        for r in results:
            a = r.aggregate
            lines.append(
                f"| {r.method} | {a.recall_at_1:.3f} | {a.recall_at_5:.3f} | "
                f"{a.recall_at_10:.3f} | {a.mrr:.3f} | {a.ndcg_at_10:.3f} |"
            )

        return "\n".join(lines)
