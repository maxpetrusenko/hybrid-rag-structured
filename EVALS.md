# Retrieval Evaluation

## Failure Taxonomy

### What Counts as Retrieval Failure?

| Failure Type | Definition | Example |
|--------------|------------|---------|
| **Miss** | Relevant document not in top-K | Query about BM25 returns docs about embeddings only |
| **Wrong Rank** | Relevant doc present but ranked too low | Correct answer at position 15 |
| **Wrong Entity** | Retrieved wrong entity with same name | "PostgreSQL vector" returns generic vector math docs |
| **Incomplete** | Retrieved partial info | Found intro but missed key technical details |

### Metrics

- **Recall@K**: Did we find the relevant doc in top K? (Binary: yes/no)
- **MRR**: How quickly did we find the first relevant? (1/rank)
- **nDCG**: Accounts for rank quality — finding relevant doc at position 1 > position 10

### Brittleness Measurement

Measured by running paraphrase queries:
- Original: "How does BM25 scoring work?"
- Paraphrase: "Explain the BM25 ranking function formula"

High variance = brittle. This project uses a single query per topic for simplicity; production systems should test paraphrase robustness.
