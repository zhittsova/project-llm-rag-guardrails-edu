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

Run the guardrail-focused Workshop 3 demo with one command:

```bash
./scripts/run_workshop3_demo.sh
```

This opens a static HTML comparison generated from the checked-in historical
400-case calibration evidence. That capture predates the evaluation-language
repair and is retained as a diagnostic, not a current result. The command makes
no API calls and keeps the frozen holdout unopened. To prepare BGE-M3 and run
the same five scenarios through the live
Fraunhofer-backed baseline and complete hybrid, use
`./scripts/run_workshop3_demo.sh --live`; that mode is the explicit approval for
remote model calls.

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
uv run guardrails-llm validate-runtime-config
```

`data/guardrail_runtime_inhouse.toml` is the versioned source for model names,
retrieval depths, evidence, classifier, policy-context, and entailment
thresholds, and runtime artifact paths. New captures include its SHA-256 hash.

Run the local instructor policy manager:

```bash
uv --directory guardrails-llm-deployment run guardrails-llm manage-policy --open
```

It edits the TOML guardrail policy through validated drafts, local simulation,
required direct/variant/benign coverage cases, atomic publish, and rollback
snapshots. It binds to localhost, makes no remote calls, and shows BGE-M3
runtime controls as read-only provenance.

The package README contains the preparation, query, capture, and evaluation
commands. The `.env` file and generated indexes/caches are local artifacts and
must not be committed.

## Evidence Status

The historical pre-repair in-house capture reached `391/400` behavior
decisions on the generated calibration split (`0.978` accuracy and macro-F1),
with no unsafe answers. The language repair changed 100 calibration prompts,
so BGE thresholds, component benchmarks, common-split scenarios, failure
analysis, and judge predictions must be rerun before those values can be called
current evidence. The historical comparison and its evidence boundary are
published in
[`reports/final_calibration_evidence.md`](guardrails-llm-deployment/reports/final_calibration_evidence.md).

The result is still not final. Expected-document citation precision is `0.752`,
below its additional `0.95` diagnostic gate, although verifier-conditioned
citation-entailment precision is `1.0`. The generated expected-document labels
need independent human review before that difference can be interpreted as a
runtime citation defect.

The frozen 400-case holdout has not been run. It requires two independent human
reviews, adjudication, judge validation, and configuration freeze first. The
package provides fail-closed commands to build calibration evidence, seal the
approved runtime configuration, and check final-run readiness. Separate
holdout reviewer files and explicit reconciliation preserve independent human
labeling before canonical annotations are written.

The LLM-judge study uses two blinded, family-disjoint 200-output sets. Qwen and
MiniMax predictions can be captured without labels, but agreement metrics are
reported only after two human reviews and disagreement adjudication. The
package includes a reviewer-isolated local UI with SQLite autosave,
question-grouped sections, dataset-issue flags, and atomic JSONL export. Start
one process per reviewer so neither reviewer can see the other's labels:

```bash
uv --directory guardrails-llm-deployment run guardrails-llm \
  review-judge-study \
  --study-dir path/to/human_judge_study \
  --reviewer reviewer_a \
  --open
```

Sidebar counters track fully labelled items immediately. An annotator ID is
still required before a completed section is exported to the reviewer JSONL.

The UI refuses to open a study that fails duplicate, language-template,
model-backed-output, or evidence-coverage checks. Technique mappings and model
predictions are never served to the browser.

For a time-constrained assisted review, create separate rubric recommendations
and expose reviewer switching explicitly:

```bash
uv --directory guardrails-llm-deployment run guardrails-llm \
  prepare-judge-recommendations --study-dir path/to/human_judge_study
uv --directory guardrails-llm-deployment run guardrails-llm \
  review-judge-study --study-dir path/to/human_judge_study \
  --reviewer reviewer_a --allow-reviewer-switch --open
```

Recommendations are hidden by default. Every reveal or copy is recorded as
assisted provenance and is not independent human ground truth. After both
human reviews are complete, `review-judge-reconciliation` shows Reviewer A,
Reviewer B, and the recommendation side by side and saves adjudications only
for human disagreements.

## Development

The import package is `guardrails_llm`; the installed command is
`guardrails-llm`. See
[`guardrails-llm-deployment/README.md`](guardrails-llm-deployment/README.md) for
technical commands and [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a PR.
