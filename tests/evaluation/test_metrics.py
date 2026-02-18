"""Tests for evaluation metrics."""

import pytest
from src.evaluation.metrics import (
    RecallAtK, MRR, NDCG, compute_metrics, AggregateMetrics
)


def test_recall_at_k_perfect():
    """Test recall when all relevant docs retrieved."""
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = {"doc1", "doc2"}
    recall = RecallAtK.compute(retrieved, relevant, 5)
    assert recall == 1.0


def test_recall_at_k_partial():
    """Test recall with partial results."""
    retrieved = ["doc1", "doc3", "doc4"]
    relevant = {"doc1", "doc2"}
    recall = RecallAtK.compute(retrieved, relevant, 5)
    assert recall == 0.5


def test_recall_at_k_cutoff():
    """Test recall respects K cutoff."""
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = {"doc3"}
    recall = RecallAtK.compute(retrieved, relevant, 2)
    assert recall == 0.0  # doc3 is at position 3


def test_mrr_first_result():
    """Test MRR when first result is relevant."""
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = {"doc1"}
    mrr = MRR.compute(retrieved, relevant)
    assert mrr == 1.0


def test_mrr_second_result():
    """Test MRR when second result is relevant."""
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = {"doc2"}
    mrr = MRR.compute(retrieved, relevant)
    assert mrr == 0.5


def test_mrr_no_relevant():
    """Test MRR when no relevant docs found."""
    retrieved = ["doc1", "doc2"]
    relevant = {"doc3"}
    mrr = MRR.compute(retrieved, relevant)
    assert mrr == 0.0


def test_ndcg_perfect():
    """Test nDCG when all relevant docs at top."""
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = {"doc1", "doc2"}
    ndcg = NDCG.compute(retrieved, relevant, 10)
    assert ndcg == 1.0


def test_ndcg_partial():
    """Test nDCG with some relevant docs."""
    retrieved = ["doc1", "doc3", "doc2"]
    relevant = {"doc1", "doc2"}
    ndcg = NDCG.compute(retrieved, relevant, 10)
    assert 0 < ndcg < 1  # Not perfect, not zero


def test_compute_metrics():
    """Test computing all metrics at once."""
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc2", "doc4"}

    metrics = compute_metrics(retrieved, relevant)

    assert "recall_at_1" in metrics
    assert "recall_at_5" in metrics
    assert "recall_at_10" in metrics
    assert "mrr" in metrics
    assert "ndcg_at_10" in metrics

    # doc2 is at position 2
    assert metrics["mrr"] == 0.5
    assert metrics["recall_at_5"] == 0.5  # 2 of 4 relevant in top 5
