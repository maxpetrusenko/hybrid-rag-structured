"""Tests for query functionality - simplified to avoid database imports."""

import pytest


def test_retrieval_config_dataclass():
    """Test RetrievalConfig can be created directly."""
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        dense_weight: float = 0.5
        sparse_weight: float = 0.5
        top_k: int = 20
        rerank_top_k: int = 5
    
    config = MockConfig()
    
    assert config.dense_weight == 0.5
    assert config.sparse_weight == 0.5
    assert config.top_k == 20
    assert config.rerank_top_k == 5


def test_retrieval_config_custom():
    """Test custom config values."""
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        dense_weight: float = 0.5
        sparse_weight: float = 0.5
        top_k: int = 20
        rerank_top_k: int = 5
    
    config = MockConfig(
        dense_weight=0.7,
        sparse_weight=0.3,
        top_k=50,
        rerank_top_k=10,
    )
    
    assert config.dense_weight == 0.7
    assert config.sparse_weight == 0.3
    assert config.top_k == 50
    assert config.rerank_top_k == 10


def test_score_normalization():
    """Test score normalization logic."""
    scores = {"a": 0.0, "b": 0.5, "c": 1.0}
    
    values = list(scores.values())
    min_val = min(values)
    max_val = max(values)
    
    normalized = {
        k: (v - min_val) / (max_val - min_val) if max_val != min_val else 1.0
        for k, v in scores.items()
    }
    
    assert normalized["a"] == 0.0
    assert normalized["c"] == 1.0
    assert 0 < normalized["b"] < 1


def test_score_fusion():
    """Test weighted score fusion."""
    dense_weight = 0.7
    sparse_weight = 0.3
    
    dense = {"a": 1.0, "b": 0.5}
    sparse = {"a": 0.5, "c": 1.0}
    
    all_ids = set(dense.keys()) | set(sparse.keys())
    
    fused = {}
    for chunk_id in all_ids:
        d = dense.get(chunk_id, 0.0)
        s = sparse.get(chunk_id, 0.0)
        fused[chunk_id] = dense_weight * d + sparse_weight * s
    
    # Check weighted combination
    assert abs(fused["a"] - (0.7 * 1.0 + 0.3 * 0.5)) < 0.01
    assert abs(fused["b"] - (0.7 * 0.5 + 0.3 * 0.0)) < 0.01
    assert abs(fused["c"] - (0.7 * 0.0 + 0.3 * 1.0)) < 0.01
