# Demo

This is a deterministic local walkthrough for the retrieval core. It uses local runtime files under `data/`; that directory is gitignored because it holds generated indexes, eval outputs, and user corpora.

## Prerequisites

- Python 3.11+
- Docker
- `uv`
- OpenAI API key for embeddings

## Run

```bash
docker compose up -d
uv pip install -e ".[dev]" asyncpg tenacity
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`.

```bash
mkdir -p data/documents data/queries

cat > data/documents/bm25.md <<'EOF'
# BM25

BM25 ranks exact term matches with term frequency, inverse document frequency, and document length normalization.
It is useful in RAG when exact keywords, identifiers, product names, or rare terms matter more than semantic similarity.
EOF

cat > data/documents/pgvector.md <<'EOF'
# pgvector

pgvector stores embedding vectors in Postgres and supports approximate or exact similarity search.
In this project, pgvector handles dense semantic retrieval while Tantivy handles sparse lexical retrieval.
EOF

cat > data/queries/queries.jsonl <<'EOF'
{"query":"How does BM25 scoring work?","relevant_docs":["bm25.md"],"category":"sparse"}
{"query":"Where are embeddings stored?","relevant_docs":["pgvector.md"],"category":"dense"}
EOF

python -m src.ingestion ingest data/documents
python -m src.retrieval "How does BM25 scoring work?"
python -m src.evaluation --queries data/queries/queries.jsonl --output data/eval_results.jsonl
```

## Expected Output Shape

Ingestion:

```text
Ingested 2 documents, 2 chunks
```

Query:

```text
Query: How does BM25 scoring work?
Found 2 results

[1] Score: 1.000 (D: 1.000, S: 1.000)
    # BM25
    BM25 ranks exact term matches...
```

Eval:

```text
# Retrieval Evaluation Results

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|--------|----------|----------|-----------|-----|---------|
| dense | ... |
| sparse | ... |
| hybrid | ... |

Results saved to data/eval_results.jsonl
```

Scores depend on the embedding model response and local corpus size. The important demo proof is that dense, sparse, and hybrid are evaluated through the same labeled query set.
