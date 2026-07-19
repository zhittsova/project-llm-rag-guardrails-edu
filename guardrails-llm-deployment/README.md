# Guardrails in LLM Deployment

Python package for implementing and evaluating guardrails in a
retrieval-augmented course assistant. Python 3.11 or newer and `uv` are
required.

## What Is Implemented

- baseline RAG without guardrails;
- normalized JSONL corpus validation and LangChain chunking;
- lexical, hashing-vector, and BGE-M3/Chroma retrieval;
- native course, visibility, and policy source-type filters;
- configurable regex, fuzzy, and embedding-similarity checks;
- optional Qwen input classification through an OpenAI-compatible API;
- BGE evidence sufficiency and policy-context retrieval gates;
- optional Qwen answer generation and entailment verification;
- verifier-selected citations, abstention, PII, injection, and academic-
  integrity handling;
- versioned evaluation splits, resumable model captures, confidence intervals,
  and compact reports.

## Structure

```text
src/guardrails_llm/
  pipeline.py              guardrailed runtime cascade
  baseline_pipeline.py     baseline RAG without guardrails
  vector.py                Chroma indexing and native-filtered retrieval
  guardrail_policy.py      TOML-backed rules and similarity policy
  openai_models.py         gated OpenAI-compatible model adapters
  model_profiles.py        local and Fraunhofer in-house profiles
  evaluation.py            behavior and grounding metrics
  e2e_capture.py           resumable common-split model experiments
data/                      corpus, policy, and versioned evaluation files
reports/                   compact calibration evidence
tests/                     deterministic tests with fake model clients
```

## Install and Verify

From this package directory:

```bash
uv sync --dev
uv run pytest
uv build
```

## Offline Workflow

No API key is needed for the default local profile:

```bash
uv run guardrails-llm validate-corpus --corpus data/course_docs.jsonl
uv run guardrails-llm build-index \
  --corpus data/course_docs.jsonl \
  --index-dir indexes/chroma
uv run guardrails-llm query \
  --mode guardrailed \
  --retriever vector \
  --index-dir indexes/chroma \
  --question "What is retrieval augmented generation?"
```

This path uses deterministic hashing embeddings and extractive answers. It is
useful for development and regression tests, but it is not the semantic
Milestone 3 configuration.

## Fraunhofer In-House Workflow

Store the endpoint and key only in the ignored `.env` file:

```text
OPENAI_API_URL=<Fraunhofer OpenAI-compatible v1 URL>
OPENAI_API_KEY=<secret>
```

Inspect configuration without making a model call:

```bash
uv run guardrails-llm model-config --profile inhouse
```

Prepare cached BGE-M3 vectors and the real-course index only after approving
remote use:

```bash
uv run guardrails-llm prepare-inhouse-bge --allow-remote-models
```

Run one complete model-backed query:

```bash
uv run guardrails-llm query \
  --profile inhouse \
  --allow-remote-models \
  --question "What is declarative knowledge?"
```

The profile uses:

- embeddings: `BAAI/bge-m3`;
- classifier, answer model, entailment verifier: `Qwen/Qwen3.6-35B-A3B`;
- retrieval: top 8 course chunks plus up to 2 native-filtered policy chunks;
- general evidence threshold: `0.5203531980514526`;
- policy-context candidate threshold: `0.51`.

Every remote command requires `--allow-remote-models`. The endpoint host,
models, prompt versions, thresholds, and input hashes are stored in manifests;
credentials and raw prompt text are not stored in embedding caches.

Use `guardrails-llm --help` and the evaluation commands below for the complete
capture sequence.

## Evaluation Status

The versioned v2 dataset has 2,000 generated cases:

- 1,200 development;
- 400 calibration;
- 400 frozen holdout.

The latest identical-split calibration comparison is in
[`reports/inhouse_common_split_calibration_v3.json`](reports/inhouse_common_split_calibration_v3.json).
The model-backed hybrid reached `0.97` behavior accuracy and `0.9699` macro-F1,
but answer recall and false-refusal gates still failed. Policy-aware retrieval
is a calibration candidate and must pass focused and full end-to-end reruns
before it is presented as an improvement.

The holdout remains unopened until double human review and adjudication are
complete. Generated labels are not treated as final human ground truth.

## Contribution Rules

Use feature branches, small conventional commits, tests, and PRs. Remote model
paths must stay explicitly gated, secrets must never be committed, and each PR
must review both READMEs and contributor documentation. The complete checklist
is in [`../CONTRIBUTING.md`](../CONTRIBUTING.md).
