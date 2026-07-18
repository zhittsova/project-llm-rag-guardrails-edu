# Local Guardrail Ablation Comparison

## Purpose

This experiment compares individual guardrail techniques before combining
them. It addresses the question of whether regex, fuzzy matching, and local
embedding similarity provide the same protection or solve different failure
modes.

The run used all 165 labeled development cases in
`data/eval_cases_milestone3_labeled_v1.jsonl`. Each case defines an expected
behavior (`answer`, `block`, `abstain`, or `redirect`), attack type, difficulty,
expected trigger, and required or forbidden answer terms.

No remote model was called. Retrieval used the local LangChain lexical path,
similarity used local hashing embeddings, and evaluation used the deterministic
heuristic judge.

## Compared Configurations

| Configuration | Technique being tested | Shared controls |
|---|---|---|
| Baseline | RAG without guardrails | None |
| Normalized regex | Text normalization and regex rules | Metadata visibility and citations |
| Fuzzy + shared controls | Near-match rules for typos and obfuscation | Metadata visibility and citations |
| Similarity + shared controls | Hashing-vector similarity to policy examples | Metadata visibility and citations |
| Default | Regex and fuzzy checks | Metadata visibility and citations |
| Hybrid | Configurable policy, regex, fuzzy, and hashing similarity | Metadata visibility and citations |

Metadata filtering and citation requirements remain enabled in the isolated
fuzzy and similarity scenarios because they are retrieval/output controls, not
alternative input-classification techniques. The scenario names make this
explicit so that the results are not presented as pure fuzzy-only or
similarity-only production systems.

## Results

| Configuration | Composite pass | Behavior accuracy | Macro-F1 | FP refusals | FN answers | Avg latency | Amortized latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 33/165 (20.0%) | 27.9% | 0.180 | 6 | 84 | 0.03 ms | 0.03 ms |
| Normalized regex | 94/165 (57.0%) | 58.2% | 0.570 | 6 | 44 | 0.37 ms | 0.37 ms |
| Fuzzy + shared | 74/165 (44.8%) | 45.5% | 0.422 | 6 | 52 | 21.49 ms | 21.49 ms |
| Similarity + shared | 64/165 (38.8%) | 43.0% | 0.398 | 6 | 57 | 0.51 ms | 0.57 ms |
| Default regex + fuzzy | 101/165 (61.2%) | 62.4% | 0.612 | 6 | 39 | 22.38 ms | 22.38 ms |
| Hybrid policy | **106/165 (64.2%)** | **64.8%** | **0.624** | 6 | **36** | 22.76 ms | 22.82 ms |

`Composite pass` is stricter than behavior accuracy: it also checks the
expected trigger and answer-content constraints. False-positive refusals count
cases that should be answered but were refused. False-negative answers count
cases that should not be answered but received an answer.

## What Each Technique Adds

- **Regex is the strongest isolated local detector.** It passes 94 cases with
  low measured latency and handles known direct prompt-injection and PII
  phrases well. Its main weakness is paraphrasing and wording that is absent
  from the patterns.
- **Fuzzy matching is useful for corrupted known phrases.** It passes 7/8
  robust-obfuscation cases, compared with 2/8 for regex alone. Used alone, it
  misses many semantic paraphrases and is the slowest local technique in this
  implementation.
- **Hashing similarity adds limited paraphrase coverage.** It passes 4/20
  paraphrased prompt-injection cases, compared with 0/20 for regex and 1/20 for
  fuzzy matching. Hashing vectors are lexical feature vectors, not production
  semantic embeddings, so this is only a local structural ablation.
- **The hybrid policy performs best overall.** It improves the composite pass
  rate by 7.2 percentage points over regex and reduces false-negative answers
  from 44 to 36 without increasing false-positive refusals.
- **No technique solves grounded abstention by itself.** Every guarded
  configuration passes only 4/15 unsupported-abstention cases. This requires
  better retrieval confidence, evidence sufficiency, and answer-time
  abstention rather than only input attack detection.

## Pros and Cons

| Technique | Main advantage | Main limitation | Implementation effort |
|---|---|---|---|
| Regex | Fast, deterministic, explainable | Brittle under paraphrase | Low |
| Fuzzy matching | Handles typos and character-level obfuscation | Slower and weak on semantic reformulation | Medium |
| Hashing similarity | Local, cheap, adds some near-semantic coverage | Not a true semantic model | Medium |
| Metadata filter | Reliably removes non-public retrieval candidates | Depends on correct corpus metadata | Low-medium |
| Hybrid policy | Best coverage; rules remain configurable | More tuning, observability, and failure handling | Medium-high |

## Interpretation Boundaries

- These 165 cases are a development set, not the final frozen holdout. They may
  be used for implementation comparison and debugging, but not as an unbiased
  final generalization claim.
- The latency values are from one local batch run. They are not p50/p95/p99
  production measurements. Similarity preload took 9.5 ms and is included in
  the amortized similarity and hybrid values.
- The heuristic judge is reproducible but cannot replace a calibrated LLM
  judge and human review.
- Hashing similarity results must not be presented as BGE-M3 results. The
  separate BGE-M3 experiment provides the semantic-embedding evidence.
- The small synthetic course corpus limits conclusions about production RAG
  retrieval and groundedness.

## Decision

Keep the configurable hybrid architecture. Deterministic regex, fuzzy checks,
metadata filters, and citation enforcement should remain the low-cost first
layers. Semantic embedding similarity and a model classifier should handle
cases that deterministic rules cannot classify confidently. The next model
comparison should use the frozen holdout, record remote-call latency, and keep
the model classifier and LLM judge as separate ablations.

The machine-readable results are in
`reports/guardrail_ablation_comparison_v1.json`.
