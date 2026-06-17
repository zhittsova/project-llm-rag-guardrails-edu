# Guardrails in LLM Deployment

[![CI](https://github.com/zhittsova/project-llm-rag-guardrails-edu/actions/workflows/ci.yml/badge.svg)](https://github.com/zhittsova/project-llm-rag-guardrails-edu/actions/workflows/ci.yml)

Prototype project for evaluating guardrails in a retrieval-augmented learning
assistant. The implementation is local and deterministic, so the full demo can
run without an LLM API key.

The project compares a baseline RAG assistant with a guardrailed assistant that
adds input checks, retrieval filtering, output checks, and simple evaluation
reporting.

## Repository Layout

```text
guardrails-llm-deployment/
  src/guardrails_llm/       assistant, retrieval, guardrail, and CLI code
  data/                     demo corpora and evaluation cases
  docs/                     workshop notes, requirements, and slide decks
  reports/                  generated CSV and HTML demo outputs
  tests/                    pytest coverage
  pyproject.toml            package and tool configuration
scripts/
  run_workshop2_demo.sh     repository-level demo entry point
```

## Quick Start

Run the Workshop 2 demo from the repository root:

```bash
./scripts/run_workshop2_demo.sh
```

Run a guardrailed query:

```bash
uv --directory guardrails-llm-deployment run guardrails-llm query \
  --mode guardrailed \
  --retriever vector \
  --index-dir indexes/python-course-chroma \
  --corpus data/python_course_docs.jsonl \
  --question "What is declarative knowledge?"
```

Run the test suite:

```bash
uv --directory guardrails-llm-deployment run pytest
```

## Project Notes

The Python import package is `guardrails_llm`; the installed console script is
`guardrails-llm`.

The vector retriever uses local deterministic hashing embeddings with a Chroma
index, which keeps the demo reproducible. The LangChain-backed path uses
LangChain document objects and recursive text splitting, while still relying on
local scoring for repeatable evaluation.

More detailed usage and implementation notes are in
[`guardrails-llm-deployment/README.md`](guardrails-llm-deployment/README.md).
