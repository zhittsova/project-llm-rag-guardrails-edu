# OpenAI-compatible model workflow

This project uses deterministic local behavior by default. Remote model
features are optional and gated so tests, demos, and evaluation sweeps cannot
make provider calls by accident.

The same code path can use either the public OpenAI API or an OpenAI-compatible
provider, for example a LiteLLM endpoint. The generic provider is selected by local environment variables. Credentials are
never committed. The fixed experiment profile retains its endpoint configuration
for reproducibility. See [the generic profile guide](openai_compatible.md).

## Local setup

Put the key in an ignored local env file:

```bash
OPENAI_API_KEY=...
```

The default path is:

```bash
guardrails-llm-deployment/.env
```

Check only safe metadata:

```bash
uv run guardrails-llm model-config
```

This prints whether the key is present, but never prints the key value.

For an OpenAI-compatible endpoint, also add one of these variables locally:

```bash
OPENAI_BASE_URL=https://provider.example/v1
```

or use the supported alias:

```bash
OPENAI_API_URL=https://provider.example/v1
```

`OPENAI_BASE_URL` wins if both are present. `model-config` reports only safe
connection metadata such as whether a base URL is present and which host it
points to.

## Default no-cost path

These commands do not call OpenAI:

```bash
uv run pytest

uv run guardrails-llm compare-guardrails \
  --retriever langchain \
  --cases data/eval_cases_milestone3.jsonl \
  --policy data/guardrail_policy.toml \
  --judge heuristic
```

By default, the project uses:

- local hashing embeddings
- local extractive answer generation
- regex + metadata + policy guardrails
- local heuristic judge

## OpenAI-gated features

Every remote model feature must include:

```bash
--allow-remote-models
```

Without that flag the CLI exits before creating a remote client or making an API
call.

Current defaults:

- embeddings: `text-embedding-3-small`
- answer generation: `gpt-5.4-mini`
- guard classifier: `gpt-5.4-nano`
- LLM judge: `gpt-5.4-nano`

When `OPENAI_BASE_URL` or `OPENAI_API_URL` is set, answer generation,
classifier, and judge calls use Chat Completions for better compatibility with
LiteLLM-style endpoints. Embeddings still use the embeddings endpoint.

Models used in the in-house runtime:

- embeddings: `BAAI/bge-m3`
- embedding fallback: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- answer generation: `Qwen/Qwen3.6-35B-A3B`
- guard classifier: `Qwen/Qwen3.6-35B-A3B`
- runtime judge default: `Qwen/Qwen3.6-35B-A3B`

The separately calibrated LLM judge uses
`MiniMaxAI/MiniMax-M2.5`. Qwen3Guard is available through the dedicated
`capture-qwen3guard-classifier` and `compare-qwen3guard-classifier` commands.
It is evaluated as a generic safety component, not as the domain classifier,
answer model, entailment verifier, or judge. See the
[`Qwen3Guard classifier comparison`](../README.md#qwen3guard-classifier-comparison)
for the full workflow and evidence boundary.

Small approved smoke tests:

```bash
uv run guardrails-llm evaluate \
  --judge openai \
  --judge-model Qwen/Qwen3.6-35B-A3B \
  --allow-remote-models \
  --limit-cases 5 \
  --cases data/eval_cases_milestone3.jsonl
```

```bash
uv run guardrails-llm query \
  --generator openai \
  --answer-model Qwen/Qwen3.6-35B-A3B \
  --allow-remote-models \
  --question "What is retrieval augmented generation?"
```

Remote embeddings rebuild the vector index, so use a separate index directory:

```bash
uv run guardrails-llm build-index \
  --embedding-provider openai \
  --embedding-model BAAI/bge-m3 \
  --allow-remote-models \
  --corpus data/python_course_docs.jsonl \
  --index-dir indexes/python-course-bge-m3
```

The vector index writes a manifest with the embedding provider/model. A query
will fail if the requested embedding provider does not match the index manifest.

## Model calibration capture

The calibration harness separates remote capture from local scoring. It writes
one replay-compatible JSONL file per component plus a manifest containing the
model, provider category, request count, prediction coverage, and p50/p95
latency. The manifest never stores the API key or full base URL.

Run the balanced classifier pilot only after approving these six remote calls.
Stratified selection chooses one case from each classifier label:

```bash
uv run guardrails-llm capture-model-calibration \
  --component classifier \
  --classifier-model Qwen/Qwen3.6-35B-A3B \
  --limit-cases 6 \
  --selection-strategy stratified \
  --allow-remote-models \
  --classifier-output reports/model_classifier_pilot_predictions.jsonl \
  --manifest-output reports/model_classifier_pilot_manifest.json
```

Review parse errors, labels, confidence, latency, and the manifest before
running the separate eight-call judge pilot. Judge selection covers matched
and mismatched actual behavior for answer, block, abstain, and redirect:

```bash
uv run guardrails-llm capture-model-calibration \
  --component judge \
  --judge-model Qwen/Qwen3.6-35B-A3B \
  --limit-cases 8 \
  --selection-strategy stratified \
  --allow-remote-models \
  --judge-output reports/judge_pilot_predictions.jsonl \
  --manifest-output reports/judge_pilot_manifest.json
```

The captured files can then be scored locally without another API call:

```bash
uv run guardrails-llm evaluate-model-calibration \
  --classifier-predictions reports/model_classifier_pilot_predictions.jsonl \
  --judge-predictions reports/judge_pilot_predictions.jsonl \
  --output-json reports/model_calibration_pilot_evaluation.json
```

The replay report counts missing or invalid responses against end-to-end
quality. A syntactically valid response with an invalid label, score, boolean,
or field set is also treated as a model error. TLS verification must remain
enabled; do not work around an untrusted certificate to run the pilot.

## Guardrail comparison

The comparison command reports baseline, deterministic guardrails, hybrid policy
guardrails, and, when requested, model-classifier guardrails:

```bash
uv run guardrails-llm compare-guardrails \
  --retriever langchain \
  --cases data/eval_cases_milestone3.jsonl \
  --policy data/guardrail_policy.toml \
  --guard-classifier openai \
  --judge openai \
  --allow-remote-models \
  --limit-cases 5
```

Use the comparison to evaluate guardrail techniques by accuracy, latency,
robustness, and implementation effort.
