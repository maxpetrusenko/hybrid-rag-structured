"""Tests for text chunking."""

import pytest
from src.ingestion.chunker import TextChunker, MarkdownChunker


def test_text_chunker_basic():
    """Test basic text chunking."""
    chunker = TextChunker(chunk_size=100, overlap=20)
    text = "a" * 300  # 300 characters
    chunks = chunker.chunk(text)

    # Should create ~3 chunks with overlap
    assert len(chunks) == 3
    assert all(c.content for c in chunks)
    assert chunks[0].index == 0
    assert chunks[1].index == 1


def test_text_chunker_with_metadata():
    """Test chunking preserves metadata."""
    chunker = TextChunker(chunk_size=50, overlap=10)
    chunks = chunker.chunk("test content", metadata={"source": "test.md"})

    assert len(chunks) == 1
    assert chunks[0].metadata == {"source": "test.md"}


def test_markdown_chunker_respects_headings():
    """Test markdown chunker splits at headings."""
    chunker = MarkdownChunker(chunk_size=100, overlap=10)
    md = """# Title

Some content here

## Section 2

More content
"""
    chunks = chunker.chunk(md)

    # Should create chunks split by headings
    assert len(chunks) >= 2
    # First chunk should have heading metadata
    assert any(c.metadata.get("heading") == "Title" for c in chunks)


def test_markdown_chunker_empty_text():
    """Test chunking empty text."""
    chunker = MarkdownChunker()
    chunks = chunker.chunk("")
    assert len(chunks) == 0
