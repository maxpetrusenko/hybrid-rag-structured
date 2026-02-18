"""Data models for documents and chunks."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column, String, Text, Float, DateTime, JSON, Integer, ForeignKey,
    create_engine, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class DocumentTable(Base):
    """Postgres table for documents."""
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    source_uri = Column(String, nullable=False, index=True)
    title = Column(String)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    chunks = relationship("ChunkTable", back_populates="document", cascade="all, delete-orphan")


class ChunkTable(Base):
    """Postgres table for chunks with vector embeddings."""
    __tablename__ = "chunks"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)  # OpenAI text-embedding-3-small
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("DocumentTable", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
    )


# Pydantic models for API
class Document(BaseModel):
    """Document model."""
    id: str
    source_uri: str
    title: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    """Chunk model with content and embedding."""
    id: str
    document_id: str
    chunk_index: int
    content: str
    embedding: Optional[list[float]] = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChunkWithScore(Chunk):
    """Chunk with retrieval score."""
    score: float
    source: str  # "dense", "sparse", or "hybrid"
