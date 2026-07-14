# BGE-M3 Embedding Experiment

## Question

Does a real semantic embedding model improve this project compared with the
local hashing embeddings?

The experiment isolates two different uses of embeddings:

1. BGE-M3 for Chroma retrieval.
2. BGE-M3 for similarity-based input guardrails.

The model was accessed through the in-house OpenAI-compatible endpoint as
`BAAI/bge-m3`. No remote answer model, classifier, or LLM judge was called.
Evaluation used the local extractive answer generator and heuristic judge.

## Setup

- Evaluation set: 165 prompts in `data/eval_cases_milestone3.jsonl`.
- Corpus: six synthetic course documents in `data/course_docs.jsonl`.
- Calibration split: 126 prompts.
- Held-out validation split: 39 prompts.
- Split rule: `blake2b(case_id, digest_size=2) % 10`; buckets below 7 are
  calibration and the rest are validation.
- BGE-M3 vector dimension observed during the smoke test: 1024.
- BGE-M3 similarity thresholds: prompt injection `0.61`, PII `0.57`, and
  academic integrity `0.62`.

The thresholds are specific to BGE-M3. Hashing thresholds and BGE-M3
thresholds are not interchangeable because their score distributions differ.

## Guardrail Results

The full 165-case result is useful as an implementation check, but it contains
the calibration cases. The held-out table is the stronger result.

| Technique, full set | Passed | Pass rate | False positives | False negatives |
|---|---:|---:|---:|---:|
| Baseline RAG | 33/165 | 20.0% | 6 | 84 |
| Normalized regex + metadata | 94/165 | 57.0% | 6 | 44 |
| Regex + fuzzy + metadata | 102/165 | 61.8% | 6 | 39 |
| Hybrid + hashing similarity | 107/165 | 64.8% | 6 | 36 |
| Hybrid + BGE-M3 similarity | 126/165 | 76.4% | 6 | 20 |

| Technique, held-out only | Passed | Pass rate | False positives | False negatives |
|---|---:|---:|---:|---:|
| Baseline RAG | 6/39 | 15.4% | 2 | 22 |
| Normalized regex + metadata | 16/39 | 41.0% | 2 | 15 |
| Regex + fuzzy + metadata | 20/39 | 51.3% | 2 | 12 |
| Hybrid + hashing similarity | 22/39 | 56.4% | 2 | 11 |
| Hybrid + BGE-M3 similarity | 26/39 | 66.7% | 1 | 6 |

On the held-out split, replacing hashing similarity with BGE-M3 improved the
hybrid policy by 4 cases and reduced false-negative answers from 11 to 6. The
largest visible gains were in PII, retrieval-access requests, and academic
integrity. Unsupported-abstention failures remained; semantic input similarity
does not solve retrieval groundedness by itself.

## Retrieval Result

The hashing-vector and BGE-M3-vector runs both passed 102 of 165 cases in the
hybrid scenario. This does **not** prove equal retrieval quality. The current
evaluation labels expected answer/refusal behavior, but it does not label the
expected document or chunk. It therefore cannot calculate Recall@k, MRR, or
nDCG and cannot prove that BGE-M3 retrieves more relevant evidence.

The correct conclusion for this milestone is:

- BGE-M3 integration works for Chroma retrieval.
- BGE-M3 retrieval improvement is not demonstrated by the current benchmark.
- BGE-M3 improves similarity guardrail detection on this held-out split.

## Latency and API Use

The full guard-similarity run embedded 173 unique texts in two provider calls.
The remote preload took about 4.84 seconds. The held-out run embedded 48 texts
in one call and took about 3.01 seconds. After preloading, per-case pipeline
latency excludes the network call; the held-out batch-amortized BGE-M3 latency
was 85.73 ms per case.

These batch numbers support offline evaluation. They are not a measurement of
uncached interactive latency. A production service should cache policy-example
vectors persistently and measure one-query p50, p95, and p99 latency.

## Limitations

- The held-out split has only 39 prompts and is not an external test set.
- The corpus is a six-document synthetic corpus, not the full course corpus.
- Some categories contain very few held-out examples.
- The heuristic judge is deterministic but not a substitute for human review
  or an independently validated LLM judge.
- The experiment measures one embedding model and one set of thresholds.
- Statistical confidence intervals and repeated robustness transformations are
  not included yet.

## Next Evidence Needed

1. Add retrieval cases with `expected_doc_ids` and report Recall@k and MRR for
   hashing versus BGE-M3 on the real course corpus.
2. Add a larger external adversarial set that was not used for threshold
   selection.
3. Measure uncached single-query latency and endpoint failure behavior.
4. Compare BGE-M3 with the available multilingual sentence-transformer model.
5. Add the model classifier and LLM-as-judge as separate ablations after
   explicit approval for LLM calls.

## Reproduction

The committed JSON artifacts contain the full summaries:

- `reports/hashing_vector_guardrail_comparison.json`
- `reports/bge_m3_retrieval_comparison.json`
- `reports/local_guardrail_comparison.json`
- `reports/bge_m3_similarity_guard_comparison.json`
- `reports/hashing_similarity_guard_validation.json`
- `reports/bge_m3_similarity_guard_validation.json`

Remote runs require the local ignored `.env` and an explicit
`--allow-remote-models` flag. API keys are not stored in these artifacts.
