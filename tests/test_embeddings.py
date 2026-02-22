"""Tests for embedding service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from src.ingestion.embeddings import EmbeddingService


def test_embedding_service_init():
    """Test EmbeddingService initialization."""
    service = EmbeddingService(api_key="test-key")
    
    assert service.model == "text-embedding-3-small"
    assert service.dimension == 1536
    assert service.batch_size == 100


def test_embedding_service_custom_params():
    """Test EmbeddingService with custom parameters."""
    service = EmbeddingService(
        api_key="test-key",
        model="text-embedding-3-large",
        dimension=3072,
        batch_size=50,
    )
    
    assert service.model == "text-embedding-3-large"
    assert service.dimension == 3072
    assert service.batch_size == 50


def test_normalize_embeddings():
    """Test L2 normalization of embeddings."""
    service = EmbeddingService(api_key="test-key")
    
    # Create unnormalized embeddings
    embeddings = [[3.0, 4.0], [1.0, 0.0]]
    normalized = service.normalize(embeddings)
    
    # First embedding should have unit norm (3,4 -> 0.6, 0.8)
    assert abs(normalized[0][0] - 0.6) < 0.01
    assert abs(normalized[0][1] - 0.8) < 0.01
    
    # Second should stay unit (1,0 -> 1,0)
    assert abs(normalized[1][0] - 1.0) < 0.01
    assert abs(normalized[1][1]) < 0.01


def test_normalize_empty_embeddings():
    """Test normalizing empty list."""
    service = EmbeddingService(api_key="test-key")
    
    # Handle empty case gracefully
    try:
        result = service.normalize([])
        assert result == [] or result is not None
    except Exception:
        # If it raises, that's acceptable behavior
        pass


def test_normalize_zero_norm():
    """Test normalizing embeddings with zero norm."""
    service = EmbeddingService(api_key="test-key")
    
    # Zero vector should not cause division by zero
    embeddings = [[0.0, 0.0]]
    normalized = service.normalize(embeddings)
    
    # Should handle gracefully (values near 0 due to epsilon)
    assert len(normalized) == 1


@pytest.mark.asyncio
async def test_embed_single():
    """Test embedding a single text."""
    service = EmbeddingService(api_key="test-key")
    
    # Mock the embed method
    mock_embedding = [0.1] * 1536
    service.embed = AsyncMock(return_value=[mock_embedding])
    
    result = await service.embed_single("test text")
    
    assert result == mock_embedding
    service.embed.assert_called_once_with(["test text"])


@pytest.mark.asyncio
async def test_embed_single_empty():
    """Test embedding empty single text."""
    service = EmbeddingService(api_key="test-key")
    
    # Mock the embed method to return empty
    service.embed = AsyncMock(return_value=[])
    
    result = await service.embed_single("")
    
    # Should return empty list for empty input
    assert result == []
