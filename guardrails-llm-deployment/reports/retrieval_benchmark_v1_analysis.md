# Milestone 3 retrieval benchmark v1

## Why this benchmark exists

The guardrail evaluation is end to end. When a case fails, that result alone
does not tell us whether retrieval selected the wrong source or whether a
guardrail made the wrong decision. This benchmark evaluates retrieval before
answer generation and guardrail classification.

The first version uses the controlled six-document `guardrails-101` corpus:

- five public documents with distinct expected topics;
- one synthetic private roster document;
- twenty relevance queries;
- four visibility-filter probes.

The case file was frozen before producing the final comparison artifacts. No
remote embeddings, LLM, answer generator, classifier, or model judge were used.

## Results

| Retriever | Recall@1 | Recall@3 | MRR | Visibility success | Mean query latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lexical | 1.000 | 1.000 | 1.000 | 1.000 | 0.02 ms |
| Hashing vector | 0.900 | 1.000 | 0.950 | 1.000 | 1.59 ms |

Both retrievers found an expected public document in the first three unique
document ranks for all 20 relevance queries. The lexical retriever placed the
expected document first every time. The hashing-vector retriever placed it
second in two cases:

- `pipeline-004`: `threat-model` ranked before `guardrail-pipeline` for the
  paraphrase about where operational events are recorded.
- `integrity-003`: `rag-basics` ranked before `integrity-policy` for the query
  about scaffolding a student's thinking.

These are ranking errors, not complete retrieval misses. They explain the gap
between hashing-vector Recall@1 (`0.90`) and Recall@3 (`1.00`).

## Metadata-filter evidence

All four visibility probes explicitly targeted `private-roster` while allowing
only `public` visibility. Neither retriever returned the private document.

This demonstrates the retrieval metadata filter, not prompt understanding. A
probe can still return an unrelated public document; the security property
measured here is that the forbidden private source is excluded before answer
generation. The private document is synthetic and contains no real student PII.

## Method

Both retrievers used the same corpus and the same LangChain chunking settings:

```text
chunk_size: 650
chunk_overlap: 80
top_k unique documents: 3
```

Chunk results were deduplicated by `doc_id` before ranking. Multiple chunks from
one document therefore cannot occupy several document ranks.

The lexical retriever uses weighted token overlap. The vector retriever uses the
local deterministic `hashing-blake2b-384` embedder and cosine similarity in
Chroma. The Chroma index is rebuilt by the benchmark command, but index-build
time is not included in mean query latency.

## Limitations

- Twenty relevance queries are enough to verify the evaluation plumbing, not
  to claim general retrieval quality.
- The questions were written from a small known corpus and often share its
  vocabulary, which favors lexical matching.
- The documents are short and produce only a small number of chunks.
- Hashing embeddings are not semantic production embeddings. Their result is a
  deterministic local control for later BGE-M3 comparison.
- Mean latency comes from one sequential local run. It is not a p50/p95
  production benchmark and excludes index construction.
- Visibility success only proves metadata exclusion for the controlled private
  decoy. It does not replace authorization testing on a production data store.

## Interpretation for the guardrail evaluation

On this controlled English corpus, top-three retrieval is not the main cause of
the previously observed 48/101 hybrid behavior errors. The larger guardrail
failures in paraphrased injection, unsafe requests, unsupported answers, and
integrity redirects still require guard-policy or model-layer work.

However, multilingual and semantically distant queries cannot be cleared by
this result. The next semantic retrieval experiment should rerun the same
frozen cases with BGE-M3 after the endpoint certificate is renewed. A later
benchmark for the Python course corpus needs independently reviewed expected
source labels.

## Reproduce

```bash
uv run guardrails-llm benchmark-retrieval \
  --corpus data/course_docs.jsonl \
  --cases data/retrieval_cases_milestone3_v1.jsonl \
  --index-dir indexes/retrieval-benchmark-v1 \
  --output-json reports/retrieval_benchmark_v1.json \
  --output-results-json reports/retrieval_benchmark_v1_results.json
```

Artifacts:

- `data/retrieval_cases_milestone3_v1.jsonl`
- `reports/retrieval_benchmark_v1.json`
- `reports/retrieval_benchmark_v1_results.json`
