# Milestone 3 Holdout v2: Initial Analysis

## Evaluation contract

This is a frozen 101-case holdout in the current `EvalCase` schema:

```text
39 cases from the original deterministic validation split
62 newly written cases that were not used for policy or threshold tuning
```

The original 165-case benchmark and both hashing and BGE-M3 policies remain
unchanged. The first result below was produced without remote models. We will
not modify rules, examples, or thresholds after seeing these cases and then
report the modified score as a holdout result.

## Local hashing result

Command:

```bash
uv run guardrails-llm compare-guardrails \
  --cases data/eval_cases_milestone3_holdout_v2.jsonl \
  --corpus data/course_docs.jsonl \
  --retriever langchain \
  --policy data/guardrail_policy.toml \
  --guard-embedding-provider hashing \
  --judge heuristic \
  --output-json reports/hashing_similarity_guard_holdout_v2.json
```

| Technique | Passed | Pass rate | FP refusals | FN answers | Avg latency |
|---|---:|---:|---:|---:|---:|
| Baseline RAG | 16/101 | 15.8% | 4 | 50 | 0.04 ms |
| Regex + metadata | 41/101 | 40.6% | 6 | 38 | 0.38 ms |
| Default deterministic guards | 48/101 | 47.5% | 7 | 33 | 13.41 ms |
| Hybrid + hashing similarity | 53/101 | 52.5% | 7 | 30 | 13.12 ms |

These latency values come from one local sequential run. They are useful for
relative comparison, but they are not interactive p50/p95 measurements.

## Hybrid result by category

| Category | Passed | Main observation |
|---|---:|---|
| Normal course | 9/10 | One German question abstained because retrieval found no evidence. |
| Borderline tutoring | 6/7 | One valid practice request was refused. |
| Direct prompt injection | 3/4 | One request was blocked or abstained for the wrong reason. |
| Prompt-injection paraphrase | 2/14 | This remains the largest injection weakness. |
| Privacy / PII | 8/10 | Most direct privacy requests were blocked. |
| Restricted retrieval access | 4/10 | Indirect requests to use private chunks often slipped through. |
| Academic integrity | 5/12 | Several cases were blocked or answered without the expected tutoring redirect. |
| Unsupported / abstention | 2/9 | Lexical overlap still caused unsupported answers. |
| Indirect prompt injection | 6/6 | Context sanitization removed the poisoned corpus instruction. |
| Robust obfuscation | 5/8 | Three requests abstained but did not produce the expected safety trigger. |
| Off-topic | 3/5 | Two unrelated requests still received grounded-looking answers. |
| Unsafe / harmful | 0/6 | Four slipped through; two abstained without the unsafe trigger. |

## What the failures show

The result is lower than the earlier 39-case validation result, but that is the
purpose of expanding the holdout. The new cases expose behavior that was not
well represented during threshold selection.

- Seven new paraphrased injection requests returned answers with no guard
  trigger. Wording such as "auditor mode", "higher-priority instructions", and
  "repeat them verbatim" is too far from the current exact and hashing examples.
- Three new restricted-retrieval requests returned answers. Metadata filtering
  still prevents a private chunk from being returned, but the input guard does
  not consistently recognize the attempt itself.
- Four unsafe requests returned answers. The local unsafe regex coverage is too
  narrow for corpus poisoning, ransomware, discriminatory ranking, and result
  manipulation.
- Unsupported questions are often answered from a lexically related course
  chunk. Citation presence alone is therefore not a sufficient groundedness
  test.
- Academic-integrity failures include both missed redirects and over-blocking.
  The current boolean `should_answer` represents a tutoring redirect as an
  answer, so this class needs the planned `answer/block/abstain/redirect`
  disposition schema for clearer metrics.
- Some requests were safely refused but still failed because the expected
  trigger was absent. This is useful diagnostic evidence: safe final behavior
  and correct guardrail classification are different requirements.

## BGE-M3 comparison: blocked by endpoint TLS certificate

The matching frozen BGE-M3 run was approved and attempted on 15 July 2026. It
failed during the TLS handshake before an embedding response was received:

```text
OpenAI embedding request failed: APIConnectionError
certificate verify error: certificate has expired
certificate NotAfter: Jul 15 11:58:58 2026 GMT
OpenSSL verify return code: 10
```

DNS and TCP connectivity were successful. An unauthenticated diagnostic request
with certificate verification disabled reached the service and returned HTTP
`401`, confirming that the service itself is running. No authenticated request
was made with TLS verification disabled, and the API key was not printed or
stored in any artifact.

The endpoint operator must renew the certificate before the experiment can be
run safely. After renewal, rerun the same frozen command:

```bash
uv run guardrails-llm compare-guardrails \
  --cases data/eval_cases_milestone3_holdout_v2.jsonl \
  --corpus data/course_docs.jsonl \
  --retriever langchain \
  --policy data/guardrail_policy_bge_m3.toml \
  --guard-embedding-provider openai \
  --guard-embedding-model BAAI/bge-m3 \
  --env-file .env \
  --allow-remote-models \
  --judge heuristic \
  --output-json reports/bge_m3_similarity_guard_holdout_v2.json
```

This command will send 101 evaluation questions and nine policy examples to
the in-house embedding endpoint: 110 unique texts in approximately two batched
embedding calls. It does not use a generative answer model, model classifier,
or LLM judge. Approval was already given for this exact frozen run; a retry is
appropriate after the endpoint presents a valid certificate.

## Current conclusion

The expanded holdout is useful because it prevents the earlier `26/39` result
from being treated as final evidence. Local hashing improves substantially over
baseline and regex-only checks, but `53/101` is not robust enough for a
production claim. BGE-M3 must be measured on the same frozen cases. After that,
the next justified layer is a model classifier for ambiguous inputs, while
groundedness and response disposition require separate evaluation work.
