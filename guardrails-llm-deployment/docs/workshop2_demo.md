# Workshop 2 Demo

This demo currently uses the synthetic six-document corpus in
`data/course_docs.jsonl`. Replace it with the collaborator corpus after handoff,
then rebuild the index and rerun evaluation.

## Baseline RAG Setup

```bash
uv run guardrails-llm validate-corpus --corpus data/course_docs.jsonl
uv run guardrails-llm build-index --corpus data/course_docs.jsonl --index-dir indexes/chroma
```

## Course-Material Q&A

```bash
uv run guardrails-llm query --mode guardrailed --retriever vector --index-dir indexes/chroma --question "What is retrieval augmented generation?"
```

Expected behavior: the assistant answers from `rag-basics` and cites the course
source.

## Unsupported Question

```bash
uv run guardrails-llm query --mode guardrailed --retriever vector --index-dir indexes/chroma --question "Which GPU model was used to train the assistant?"
```

Expected behavior: the assistant abstains because the corpus has no supporting
course evidence.

## Failure Analysis Run

```bash
uv run guardrails-llm evaluate --mode baseline --retriever vector --index-dir indexes/chroma --output-csv reports/baseline_vector_results.csv
uv run guardrails-llm evaluate --mode guardrailed --retriever vector --index-dir indexes/chroma --output-csv reports/guardrailed_vector_results.csv
```

Current synthetic-corpus results:

- Baseline vector: `3/12`.
- Guardrailed vector: `12/12`.

The real Workshop 2 demo should rerun these commands after the collaborator
corpus arrives.
