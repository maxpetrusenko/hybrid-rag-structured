"""Tests for hybrid retrieval."""

import pytest
from src.retrieval.hybrid import RetrievalConfig, HybridRetriever, DenseRetriever


def test_retrieval_config_defaults():
    """Test default config values."""
    config = RetrievalConfig()
    assert config.dense_weight == 0.5
    assert config.sparse_weight == 0.5
    assert config.top_k == 20
    assert config.rerank_top_k == 5


def test_retrieval_config_from_env(monkeypatch):
    """Test loading config from environment."""
    monkeypatch.setenv("DENSE_WEIGHT", "0.7")
    monkeypatch.setenv("SPARSE_WEIGHT", "0.3")
    monkeypatch.setenv("TOP_K", "50")

    config = RetrievalConfig.from_env()
    assert config.dense_weight == 0.7
    assert config.sparse_weight == 0.3
    assert config.top_k == 50


def test_score_normalization():
    """Test score normalization logic."""
    retriever = HybridRetriever(None, None)

    scores = {"a": 0.1, "b": 0.5, "c": 1.0}
    normalized = retriever._normalize_scores(scores)

    assert normalized["a"] == 0.0
    assert normalized["c"] == 1.0
    assert 0 < normalized["b"] < 1


def test_score_fusion():
    """Test weighted score fusion."""
    config = RetrievalConfig(dense_weight=0.7, sparse_weight=0.3)
    retriever = HybridRetriever(None, None, config)

    dense = {"a": 1.0, "b": 0.5}
    sparse = {"a": 0.5, "c": 1.0}

    fused = retriever._fuse_scores(dense, sparse)

    assert abs(fused["a"] - (0.7 * 1.0 + 0.3 * 0.5)) < 0.01
    assert abs(fused["b"] - (0.7 * 0.5 + 0.3 * 0.0)) < 0.01
    assert abs(fused["c"] - (0.7 * 0.0 + 0.3 * 1.0)) < 0.01
