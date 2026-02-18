# hybrid-rag-structured

Hybrid RAG combining semantic (dense) + structured (sparse) retrieval over Postgres + pgvector.

## Approach

- **Python**: ingestion + retrieval experiments
- **Postgres + pgvector**: unified storage for vectors + structured data
- **BM25**: sparse retrieval via tantivy
- **Optional UI**: minimal CLI or Next.js frontend

## Why Hybrid?

Dense retrieval (embeddings) excels at semantic matching but misses exact terms. Sparse retrieval (BM25) catches keywords but fails on synonyms. Combining both = better recall and precision.

## Architecture

```
┌─────────────┐     ┌─────────────┐
│  Documents  │────>│ Ingestion   │
└─────────────┘     └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Postgres  │
                    │  + pgvector │
                    └──────┬──────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
     ┌─────────────┐              ┌─────────────┐
     │ Dense Search│              │ Sparse (BM25)│
     │ (embedding) │              │   (keyword) │
     └──────┬──────┘              └──────┬──────┘
            │                            │
            └────────────┬───────────────┘
                         ▼
                   ┌─────────────┐
                   │  Rerank +   │
                   │  Combine    │
                   └──────┬──────┘
                          ▼
                   ┌─────────────┐
                   │   Generate  │
                   └─────────────┘
```

## Quick Start

```bash
# Docker: Postgres + pgvector
docker compose up -d

# Install deps
uv pip install -e .

# Ingest documents
python -m src.ingestion ingest ./data

# Query (hybrid)
python -m src.retrieval query "your question here"
```

## Sources

- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [Tantivy Python bindings](https://github.com/quickwit-oss/tantivy-py)
- [Hybrid Retrieval Best Practices](https://arxiv.org/abs/2401.16487)
