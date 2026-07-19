# Guardrails in LLM Deployment

[![CI](https://github.com/zhittsova/project-llm-rag-guardrails-edu/actions/workflows/ci.yml/badge.svg)](https://github.com/zhittsova/project-llm-rag-guardrails-edu/actions/workflows/ci.yml)

Prototype for comparing guardrail techniques in a retrieval-augmented learning
assistant. The project includes an offline deterministic profile for local
development and an explicitly gated in-house profile for model-backed Milestone
3 experiments.

The current comparison covers baseline RAG, regex, fuzzy matching, BGE-M3
similarity, metadata filtering, Qwen classification, grounded answer
generation, entailment verification, citation filtering, and abstention.

## Runtime Flow

```text
user request
  -> deterministic input checks
  -> BGE-M3 similarity checks
  -> Qwen classifier for unresolved requests
  -> Chroma retrieval with course/visibility filters
  -> BGE evidence and policy-context gates
  -> Qwen answer generation
  -> Qwen entailment verification
  -> verifier-approved citations or abstention
  -> output PII/injection checks
```

The local profile uses hashing embeddings and extractive answers so tests and
the Workshop 2 demo do not require credentials. Hashing is an offline fallback,
not the Milestone 3 semantic-retrieval result.

## Repository Layout

```text
guardrails-llm-deployment/
  src/guardrails_llm/   pipeline, retrieval, guardrails, models, and evaluation
  data/                 normalized corpora, policies, and versioned eval cases
  docs/                 technical workflows and corpus contract
  reports/              checked-in compact evaluation evidence
  tests/                pytest coverage
  pyproject.toml        uv package configuration
scripts/                repository-level demo entry points
Workshop1..3/           local workshop planning and presentation material
```

## Local Quick Start

```bash
uv --directory guardrails-llm-deployment sync --dev
uv --directory guardrails-llm-deployment run pytest
./scripts/run_workshop2_demo.sh
```

Ask an offline guardrailed question:

```bash
uv --directory guardrails-llm-deployment run guardrails-llm query \
  --mode guardrailed \
  --retriever langchain \
  --question "What is retrieval augmented generation?"
```

## In-House Profile

The `inhouse` profile is locked to the configured Fraunhofer OpenAI-compatible
endpoint. It uses `BAAI/bge-m3` for embeddings and
`Qwen/Qwen3.6-35B-A3B` for classification, answers, and entailment. It refuses
remote calls unless `--allow-remote-models` is present.

From `guardrails-llm-deployment/`, first inspect credential-safe configuration:

```bash
uv run guardrails-llm model-config --profile inhouse
```

The package README contains the preparation, query, capture, and evaluation
commands. The `.env` file and generated indexes/caches are local artifacts and
must not be committed.

## Evidence Status

The complete in-house hybrid reached `388/400` behavior decisions on the
generated calibration split (`0.97` accuracy, `0.9699` macro-F1), with no unsafe
answers. Answer recall (`0.88`) and safe false-refusal rate (`0.06`) still missed
their gates, so this is calibration evidence rather than a final result.

The frozen 400-case holdout has not been run. It requires two independent human
reviews, adjudication, and configuration freeze first.

## Development

The import package is `guardrails_llm`; the installed command is
`guardrails-llm`. See
[`guardrails-llm-deployment/README.md`](guardrails-llm-deployment/README.md) for
technical commands and [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a PR.
