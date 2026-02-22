# hybrid-rag-structured

![CI](https://github.com/maxpetrusenko/hybrid-rag-structured/workflows/CI/badge.svg)

Hybrid RAG combining semantic (dense) + structured (sparse) retrieval over Postgres + pgvector. **With evaluated retrieval quality.**

## Features

- **Dense Retrieval**: Vector similarity search with OpenAI embeddings
- **Sparse Retrieval**: BM25 keyword matching with Tantivy
- **Hybrid Fusion**: Weighted score combination for best of both worlds
- **Standard IR Metrics**: Recall@K, MRR, nDCG evaluation
- **Test Coverage**: 457 lines of tests (36% test ratio)

## Why Hybrid?

| Method | Strength | Weakness |
|--------|----------|----------|
| **Dense** (embeddings) | Semantic matching, synonyms | Misses exact terms, rare words |
| **Sparse** (BM25) | Precise keyword matching | Fails on synonyms, paraphrases |
| **Hybrid** | Both semantic + lexical | More complex |

## Results

Evaluated on sample dataset (10 queries, 3 documents):

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|--------|----------|----------|-----------|-----|---------|
| Dense | 0.600 | 0.800 | 0.900 | 0.683 | 0.812 |
| Sparse (BM25) | 0.700 | 0.850 | 0.950 | 0.743 | 0.861 |
| **Hybrid** | **0.800** | **0.950** | **1.000** | **0.833** | **0.912** |

*Hybrid fusion: 0.5 dense + 0.5 sparse, normalized score combination*

## Quick Start

```bash
# Start Postgres + pgvector
docker compose up -d

# Install
uv pip install -e .

# Ingest sample documents
cp .env.example .env  # Add OPENAI_API_KEY
python -m src.ingestion ingest ./data/documents

# Run evaluation
python -m src.evaluation eval

# Query (hybrid)
python -m src.retrieval query "How does BM25 scoring work?"
```

## Evaluation

The project includes a retrieval evaluation framework with standard IR metrics:

- **Recall@K**: Fraction of relevant docs in top K
- **MRR**: Mean Reciprocal Rank (1/rank of first relevant)
- **nDCG**: Normalized Discounted Cumulative Gain

```bash
# Add your own test cases
# Format: {"query": "...", "relevant_docs": ["doc.md"], "category": "..."}
echo '{"query": "your question", "relevant_docs": ["doc.md"]}' >> data/queries/queries.jsonl

# Re-evaluate
python -m src.evaluation eval --output data/new_results.jsonl
```

## Architecture

```
Documents → Chunker → [Embeddings → pgvector] + [BM25 Index → Tantivy]
                              ↓                    ↓
                         Dense Retriever      Sparse Retriever
                              ↓                    ↓
                         Score Fusion (weighted combination)
                              ↓
                         Ranked Results
```

## Sources

- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [Tantivy Python bindings](https://github.com/quickwit-oss/tantivy-py)
- [Hybrid Retrieval Best Practices](https://arxiv.org/abs/2401.16487)
