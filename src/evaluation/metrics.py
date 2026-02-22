"""Retrieval quality metrics."""

from dataclasses import dataclass
from typing import List, Set
import math


@dataclass
class EvaluationResult:
    """Result of a single query evaluation."""
    query: str
    retrieved_docs: List[str]
    relevant_docs: Set[str]
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float


@dataclass
class AggregateMetrics:
    """Aggregated metrics across multiple queries."""
    num_queries: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float

    def format(self) -> str:
        """Format as table string."""
        return f"""
# Retrieval Evaluation Results

| Metric | Value |
|--------|-------|
| Queries | {self.num_queries} |
| Recall@1 | {self.recall_at_1:.3f} |
| Recall@5 | {self.recall_at_5:.3f} |
| Recall@10 | {self.recall_at_10:.3f} |
| MRR | {self.mrr:.3f} |
| nDCG@10 | {self.ndcg_at_10:.3f} |
"""


class RecallAtK:
    """Recall@K: fraction of relevant docs found in top K results."""

    @staticmethod
    def compute(
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """Compute recall at K."""
        retrieved_at_k = set(retrieved[:k])
        if not relevant:
            return 0.0
        return len(retrieved_at_k & relevant) / len(relevant)


class MRR:
    """Mean Reciprocal Rank: average of 1/rank for first relevant result."""

    @staticmethod
    def compute(retrieved: List[str], relevant: Set[str]) -> float:
        """Compute MRR."""
        if not relevant:
            return 0.0

        for i, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                return 1.0 / i
        return 0.0


class NDCG:
    """Normalized Discounted Cumulative Gain."""

    @staticmethod
    def compute(
        retrieved: List[str],
        relevant: Set[str],
        k: int = 10
    ) -> float:
        """Compute nDCG@K.

        Assumes binary relevance (1 if relevant, 0 otherwise).
        """
        # DCG: sum of relevance / log2(rank + 1)
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k], start=1):
            relevance = 1.0 if doc in relevant else 0.0
            dcg += relevance / math.log2(i + 1)

        # IDCG: ideal DCG (all relevant docs at top)
        idcg = 0.0
        for i in range(1, min(k, len(relevant)) + 1):
            idcg += 1.0 / math.log2(i + 1)

        if idcg == 0:
            return 0.0
        return dcg / idcg


def compute_metrics(
    retrieved: List[str],
    relevant: Set[str],
) -> dict:
    """Compute all metrics for a single query."""
    return {
        "recall_at_1": RecallAtK.compute(retrieved, relevant, 1),
        "recall_at_5": RecallAtK.compute(retrieved, relevant, 5),
        "recall_at_10": RecallAtK.compute(retrieved, relevant, 10),
        "mrr": MRR.compute(retrieved, relevant),
        "ndcg_at_10": NDCG.compute(retrieved, relevant, 10),
    }
