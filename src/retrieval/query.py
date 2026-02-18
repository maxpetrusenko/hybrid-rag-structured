"""Query CLI for hybrid retrieval."""

import asyncio
import click
from dotenv import load_dotenv

from ..storage.database import Database, get_db
from ..ingestion.bm25_index import BM25Index
from ..ingestion.embeddings import EmbeddingService
from .hybrid import HybridRetriever, RetrievalConfig

load_dotenv()


@click.command()
@click.argument("query")
@click.option("--dense-weight", default=0.5, help="Weight for dense retrieval")
@click.option("--sparse-weight", default=0.5, help="Weight for sparse retrieval")
@click.option("--top-k", default=20, help="Number of results to retrieve")
@click.option("--rerank-top-k", default=5, help="Number of results after reranking")
def query_command(query: str, dense_weight: float, sparse_weight: float, top_k: int, rerank_top_k: int):
    """Query the hybrid RAG system."""
    async def run():
        await Database.init_db()

        # Setup
        config = RetrievalConfig(
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
        )
        embedder = EmbeddingService()
        bm25 = BM25Index()
        bm25.open_index()

        async with Database.get_session() as session:
            retriever = HybridRetriever(session, bm25, config)

            # Embed query
            query_embedding = await embedder.embed_single(query)

            # Retrieve
            results = await retriever.retrieve(query, query_embedding)

            # Display results
            click.echo(f"\nQuery: {query}")
            click.echo(f"Found {len(results)} results\n")

            for i, result in enumerate(results, 1):
                click.echo(f"[{i}] Score: {result.score:.3f} (D: {result.dense_score:.3f}, S: {result.sparse_score:.3f})")
                click.echo(f"    {result.content[:200]}...\n")

        await Database.close()

    asyncio.run(run())


if __name__ == "__main__":
    query_command()
