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
  final_evidence.py        calibration packaging and final-run release gates
  holdout_review.py        blinded double review and adjudication workflow
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

The latest identical-split calibration evidence is in
[`reports/inhouse_retrieval_recovery_calibration_v4.json`](reports/inhouse_retrieval_recovery_calibration_v4.json).
The model-backed hybrid reached `391/400`, `0.978` behavior accuracy and
macro-F1, `0.91` answer recall, and `0.045` false-refusal rate. It produced no
unsafe answers, and verifier-conditioned citation-entailment precision was
`1.0`.

The additional expected-document citation diagnostic remains below its gate at
`0.752`. Generated expected-document labels can omit other valid supporting
documents, so this failure remains visible and requires independent human
review rather than being silently reclassified as success.

The holdout remains unopened until double human review and adjudication are
complete. Generated labels are not treated as final human ground truth.

### Human-Calibrated LLM Judge

Prepare two blinded, family-disjoint 200-output annotation sets from a common
calibration result file:

```bash
uv run guardrails-llm prepare-judge-study \
  --source-results path/to/common_split_results.json \
  --output-dir path/to/human_judge_study
```

Check reviewer progress and reconcile two completed reviews locally:

```bash
uv run guardrails-llm judge-study-status \
  --study-dir path/to/human_judge_study
uv run guardrails-llm reconcile-judge-study \
  --study-dir path/to/human_judge_study
uv run guardrails-llm finalize-judge-study \
  --study-dir path/to/human_judge_study
```

Judge predictions are captured without reading human labels. Repeat the command
with `Qwen/Qwen3.6-35B-A3B` and `MiniMaxAI/MiniMax-M2.5`, using separate output
and manifest paths:

```bash
uv run guardrails-llm capture-judge-study \
  --study-dir path/to/human_judge_study \
  --source-results path/to/common_split_results.json \
  --judge-model Qwen/Qwen3.6-35B-A3B \
  --output path/to/judge_qwen_predictions.jsonl \
  --manifest path/to/judge_qwen_manifest.json \
  --max-concurrency 4 \
  --allow-remote-models
```

After adjudication, compare both prediction files with `evaluate-judge-study`.
The report keeps judge calibration and judge validation separate and applies
the structured-validity, per-dimension, exact-match, and groundedness gates.

### Independent Frozen-Holdout Review

Do not start this workflow until the two human reviewers are assigned. Prepare
separate blinded reviewer files without exposing generated expected labels:

```bash
uv run guardrails-llm prepare-holdout-review \
  --output-dir path/to/holdout_review
```

Each reviewer edits only their assigned `holdout_reviewer_*.jsonl` file. Check
completion, then create the agreement report and disagreement-only
adjudication file:

```bash
uv run guardrails-llm holdout-review-status \
  --study-dir path/to/holdout_review
uv run guardrails-llm reconcile-holdout-review \
  --study-dir path/to/holdout_review
```

After every disagreement is adjudicated, compile the canonical annotation
schema. Replacing the tracked annotation template requires the explicit
`--replace` flag:

```bash
uv run guardrails-llm finalize-holdout-review \
  --study-dir path/to/holdout_review \
  --output-annotations data/eval_cases_milestone3_v2_holdout_annotations.jsonl \
  --replace
uv run python scripts/finalize_eval_dataset_milestone3_v2.py \
  --corpus data/python_course_docs.jsonl
```

Preparation fingerprints the frozen source and blinded items. Reconciliation
requires complete files with distinct reviewer identities. Matching labels are
recorded as reviewer consensus; only disagreements require a separate human
adjudication. Dataset sealing still validates all 400 canonical annotations
and updates the dataset manifest before evaluation is permitted.

### Final Evidence and Holdout Release Gate

Build the consolidated seven-technique calibration report from the checked-in
400-case common-split evidence:

```bash
uv run guardrails-llm build-final-evidence
```

This writes `reports/final_calibration_evidence.json` and
`reports/final_calibration_evidence.md`. It rejects holdout-derived or
incomplete inputs and keeps failed diagnostics visible.

Only after holdout annotations are independently reviewed, adjudicated, and
sealed, freeze the exact artifacts selected during calibration:

```bash
uv run guardrails-llm seal-final-config \
  --dataset-manifest path/to/dataset_manifest.json \
  --calibration-report reports/final_calibration_evidence.json \
  --policy data/guardrail_policy.toml \
  --course-corpus data/python_course_docs.jsonl \
  --index-manifest path/to/bge_index_manifest.json \
  --output-json path/to/final_configuration.json
```

The configuration manifest records SHA-256 hashes for the dataset manifest,
calibration evidence, policy, corpus, and index manifest. It does not include
credentials.

Before the one-time frozen-holdout run, verify every human, judge, calibration,
and configuration gate:

```bash
uv run guardrails-llm check-final-readiness \
  --dataset-manifest path/to/dataset_manifest.json \
  --judge-report path/to/judge_evaluation.json \
  --selected-judge-model Qwen/Qwen3.6-35B-A3B \
  --calibration-report reports/final_calibration_evidence.json \
  --configuration-manifest path/to/final_configuration.json \
  --output-json path/to/final_readiness.json
```

The readiness command exits nonzero and lists every failed condition until all
required evidence is present. It does not execute the holdout itself.

## Contribution Rules

Use feature branches, small conventional commits, tests, and PRs. Remote model
paths must stay explicitly gated, secrets must never be committed, and each PR
must review both READMEs and contributor documentation. The complete checklist
is in [`../CONTRIBUTING.md`](../CONTRIBUTING.md).
