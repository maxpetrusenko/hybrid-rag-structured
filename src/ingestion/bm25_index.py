"""BM25 sparse retrieval index using Tantivy."""

import os
from typing import List, Dict, Tuple
from pathlib import Path
import tantivy
from dataclasses import dataclass


@dataclass
class BM25Result:
    """BM25 search result."""
    chunk_id: str
    score: float


class BM25Index:
    """Tantivy-based BM25 index for sparse retrieval."""

    def __init__(self, index_path: str | Path = "./data/bm25_index"):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.index: tantivy.Index | None = None
        self.reader: tantivy.IndexReader | None = None
        self.searcher: tantivy.Searcher | None = None

    def build_schema(self) -> tantivy.SchemaBuilder:
        """Build Tantivy schema for chunk indexing."""
        return (
            tantivy.SchemaBuilder()
            .add_text_field("content", stored=True, tokenizer_name="en_stem")
            .add_text_field("title", stored=True, tokenizer_name="en_stem")
            .add_text_field("chunk_id", stored=True, indexed=True)
            .add_integer_field("chunk_index", stored=True)
        )

    def create_index(self) -> None:
        """Create a new Tantivy index."""
        schema = self.build_schema()
        self.index = schema.create()
        self.writer = self.index.writer()

    def open_index(self) -> None:
        """Open existing Tantivy index."""
        try:
            self.index = tantivy.Index.open(str(self.index_path))
            self.reader = self.index.reader()
            self.searcher = self.reader.searcher()
        except Exception:
            # Index doesn't exist, create it
            self.create_index()
            self.index.searcher()

    def add_chunks(self, chunks: List[Tuple[str, str, str, int]]) -> None:
        """Add chunks to the index.

        Args:
            chunks: List of (chunk_id, title, content, chunk_index)
        """
        if not self.index:
            self.create_index()

        writer = self.index.writer()

        for chunk_id, title, content, chunk_index in chunks:
            doc = tantivy.Document(
                chunk_id=chunk_id,
                title=title or "",
                content=content,
                chunk_index=chunk_index,
            )
            writer.add_document(doc)

        writer.commit()
        # Reload reader
        self.reader = self.index.reader()
        self.searcher = self.reader.searcher()

    def search(
        self,
        query: str,
        limit: int = 20,
       _bm25_b: float = 0.75,
        _bm25_k1: float = 1.2,
    ) -> List[BM25Result]:
        """Search the BM25 index."""
        if not self.searcher:
            return []

        # Parse query
        query_parser = tantivy.QueryParser.for_index(
            self.index,
            ["title", "content"],
        )
        parsed_query = query_parser.parse_query(query)

        # Search
        top_docs = self.searcher.search(parsed_query, limit=limit)

        results = []
        for score, doc_address in top_docs.hits:
            doc = self.searcher.doc(doc_address)
            chunk_id = doc.get("chunk_id")[0] if doc.get("chunk_id") else ""
            results.append(BM25Result(chunk_id=chunk_id, score=score))

        return results

    def delete_index(self) -> None:
        """Delete the index."""
        import shutil
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
        self.index = None
        self.searcher = None
