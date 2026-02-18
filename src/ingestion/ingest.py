"""Ingestion pipeline: read docs, chunk, embed, index."""

import asyncio
import uuid
from pathlib import Path
from typing import List
import click

from ..storage.database import Database, get_db
from ..storage.models import DocumentTable, ChunkTable
from .chunker import MarkdownChunker, read_documents, Chunk
from .embeddings import EmbeddingService
from .bm25_index import BM25Index


class IngestionPipeline:
    """End-to-end ingestion pipeline."""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.chunker = MarkdownChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedder = EmbeddingService(model=embedding_model)
        self.bm25 = BM25Index()
        self.bm25.open_index()

    async def ingest(self, path: str | Path) -> dict:
        """Ingest documents from path.

        Returns summary with counts.
        """
        # Read documents
        docs = read_documents(path)

        if not docs:
            return {"documents": 0, "chunks": 0}

        async with Database.get_session() as session:
            all_chunks = []

            for source_uri, content in docs:
                # Create document record
                doc_id = str(uuid.uuid4())
                title = Path(source_uri).stem
                document = DocumentTable(
                    id=doc_id,
                    source_uri=source_uri,
                    title=title,
                    metadata={},
                )
                session.add(document)

                # Chunk document
                chunks = self.chunker.chunk(content, metadata={"source_uri": source_uri})

                # Prepare for batch embedding
                chunk_texts = [c.content for c in chunks]
                embeddings = await self.embedder.embed(chunk_texts)
                normalized = self.embedder.normalize(embeddings)

                # Create chunk records
                for chunk, embedding in zip(chunks, normalized):
                    chunk_id = str(uuid.uuid4())
                    chunk_record = ChunkTable(
                        id=chunk_id,
                        document_id=doc_id,
                        chunk_index=chunk.index,
                        content=chunk.content,
                        embedding=embedding,
                        metadata=chunk.metadata,
                    )
                    session.add(chunk_record)
                    all_chunks.append((chunk_id, title, chunk.content, chunk.index))

            await session.commit()

        # Build BM25 index
        self.bm25.add_chunks(all_chunks)

        return {
            "documents": len(docs),
            "chunks": len(all_chunks),
        }


@click.group()
def ingest_cli():
    """Ingest documents into the RAG system."""
    pass


@ingest_cli.command("ingest")
@click.argument("path", type=click.Path(exists=True))
@click.option("--chunk-size", default=500, help="Size of chunks in characters")
@click.option("--overlap", default=50, help="Overlap between chunks")
def ingest_command(path: str, chunk_size: int, overlap: int):
    """Ingest documents from PATH."""
    async def run():
        await Database.init_db()
        pipeline = IngestionPipeline(chunk_size=chunk_size, overlap=overlap)
        result = await pipeline.ingest(path)
        click.echo(f"Ingested {result['documents']} documents, {result['chunks']} chunks")
        await Database.close()

    asyncio.run(run())


if __name__ == "__main__":
    ingest_cli()
