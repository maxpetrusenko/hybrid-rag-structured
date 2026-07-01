# Retrieval Evaluation

This repo evaluates retrieval quality before any answer generation. The runner compares dense pgvector search, sparse BM25 search, and hybrid fusion against labeled relevant documents.

## Command

```bash
python -m src.evaluation --queries data/queries/queries.jsonl --output data/eval_results.jsonl
```

Query file format:

```json
{"query":"How does BM25 scoring work?","relevant_docs":["bm25.md"],"category":"sparse"}
```

The evaluator maps retrieved chunk IDs back to source filenames through chunk metadata, then computes aggregate metrics per retrieval method.

## Metrics

| Metric | Implementation | Signal |
| --- | --- | --- |
| Recall@1 | Relevant docs found in the first result | Best-answer precision. |
| Recall@5 | Relevant docs found in first five results | Practical context-window recall. |
| Recall@10 | Relevant docs found in first ten results | Broad retriever coverage. |
| MRR | `1 / rank` for the first relevant document | How quickly useful evidence appears. |
| nDCG@10 | Rank-discounted relevance gain | Whether relevant docs are high enough to matter. |

Reference benchmark snapshot:

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.600 | 0.800 | 0.900 | 0.683 | 0.812 |
| Sparse BM25 | 0.700 | 0.850 | 0.950 | 0.743 | 0.861 |
| Hybrid | 0.800 | 0.950 | 1.000 | 0.833 | 0.912 |

Suggested regression gate for a committed benchmark set:

| Gate | Threshold |
| --- | ---: |
| Hybrid Recall@5 | >= 0.900 |
| Hybrid MRR | >= 0.800 |
| Hybrid nDCG@10 | >= 0.880 |
| Hybrid below both dense and sparse on same metric | fail |

## Failure Taxonomy

| Failure Type | Definition | Example |
| --- | --- | --- |
| Miss | Relevant document absent from top K | Query about BM25 returns only embedding docs. |
| Wrong rank | Relevant document present but ranked too low | Correct source appears at rank 9 for a top 5 context budget. |
| Wrong entity | Same term, wrong meaning or object | "vector" retrieves math notes instead of pgvector storage notes. |
| Incomplete evidence | Partial source found, key detail absent | Intro chunk retrieved but chunk with formula/config missed. |
| Fusion regression | Hybrid underperforms one or both component retrievers | Dense finds paraphrase, BM25 noise dominates final rank. |

## Diagnostic Loop

1. Run dense, sparse, and hybrid on the same query set.
2. Identify whether the failure is semantic, lexical, chunk boundary, or fusion-weight related.
3. Tune one variable at a time: chunk size, overlap, `DENSE_WEIGHT`, `SPARSE_WEIGHT`, `TOP_K`, or `RERANK_TOP_K`.
4. Re-run `python -m src.evaluation --queries ...` and compare aggregate plus per-query JSONL rows.
5. Promote only changes that improve hybrid without hiding a dense-only or sparse-only regression.

## Current Scope

The project measures retrieval quality only. It does not yet score answer faithfulness, citation coverage, latency, token cost, or tool-call accuracy because there is no generation or agent layer in this repo.
