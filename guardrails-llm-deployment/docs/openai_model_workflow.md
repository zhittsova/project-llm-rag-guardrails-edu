# OpenAI model workflow

This project keeps local deterministic behavior as the default. OpenAI-backed
features are optional and gated so tests, demos, and evaluation sweeps do not
spend API credits by accident.

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

Every OpenAI feature must include:

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

Small approved smoke tests:

```bash
uv run guardrails-llm evaluate \
  --judge openai \
  --allow-remote-models \
  --limit-cases 5 \
  --cases data/eval_cases_milestone3.jsonl
```

```bash
uv run guardrails-llm query \
  --generator openai \
  --allow-remote-models \
  --question "What is retrieval augmented generation?"
```

OpenAI embeddings rebuild the vector index, so use a separate index directory:

```bash
uv run guardrails-llm build-index \
  --embedding-provider openai \
  --allow-remote-models \
  --corpus data/python_course_docs.jsonl \
  --index-dir indexes/python-course-openai
```

The vector index writes a manifest with the embedding provider/model. A query
will fail if the requested embedding provider does not match the index manifest.

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

This is the intended Milestone 3 direction: compare guardrail techniques by
accuracy, latency, robustness, and implementation effort.
