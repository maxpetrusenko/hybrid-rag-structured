"""Tests for BM25 index functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path

# Skip all tests if tantivy is not available
pytest.importorskip("tantivy")

from src.ingestion.bm25_index import BM25Index, BM25Result


@pytest.fixture
def temp_index_path():
    """Create a temporary directory for the index."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_bm25_result_model():
    """Test BM25Result dataclass."""
    result = BM25Result(chunk_id="test-1", score=1.5)
    assert result.chunk_id == "test-1"
    assert result.score == 1.5


def test_bm25_index_path_creation():
    """Test that index path is created if it doesn't exist."""
    temp_dir = tempfile.mkdtemp()
    index_path = Path(temp_dir) / "new_subdir" / "index"
    
    try:
        index = BM25Index(index_path=index_path)
        assert index_path.parent.exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_bm25_index_build_schema():
    """Test schema builder exists."""
    index = BM25Index()
    
    # Schema builder method should exist
    assert hasattr(index, 'build_schema')


def test_bm25_index_search_empty():
    """Test searching on uninitialized index."""
    index = BM25Index()
    
    # Should return empty list if no searcher
    results = index.search("test")
    assert results == []


def test_bm25_index_delete(temp_index_path):
    """Test deleting the index."""
    index = BM25Index(index_path=temp_index_path)
    
    # Delete should work even if index doesn't exist
    index.delete_index()
    
    assert index.index is None
    assert not Path(temp_index_path).exists()
