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
  review_store.py          SQLite drafts and atomic reviewer-file export
  review_server.py         reviewer-isolated local annotation interface
  review_recommendations.py separate, rationalized review hints
  review_reconciliation.py three-way comparison and adjudication storage
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

The historical identical-split calibration evidence is in
[`reports/inhouse_retrieval_recovery_calibration_v4.json`](reports/inhouse_retrieval_recovery_calibration_v4.json).
Before the evaluation-language repair, the model-backed hybrid reached
`391/400`, `0.978` behavior accuracy and
macro-F1, `0.91` answer recall, and `0.045` false-refusal rate. It produced no
unsafe answers, and verifier-conditioned citation-entailment precision was
`1.0`. The repair changed 100 calibration prompts and 300 development
prompts, so this is a historical diagnostic. The model-backed development and
calibration sequence must be rerun before publishing current figures.

The additional expected-document citation diagnostic remains below its gate at
`0.752`. Generated expected-document labels can omit other valid supporting
documents, so this failure remains visible and requires independent human
review rather than being silently reclassified as success.

The holdout remains unopened until double human review and adjudication are
complete. Generated labels are not treated as final human ground truth.

## Workshop 3 Demo

Run the complete offline demo from the repository root:

```bash
./scripts/run_workshop3_demo.sh
```

The command opens `reports/workshop3_guardrail_demo.html`. It uses the
checked-in 400-case calibration evidence and makes no remote calls. The report
shows the complete runtime cascade, the identical-split technique comparison,
five baseline-versus-hybrid failure scenarios, and the nine remaining false
abstentions. It labels the evidence as calibration-only and does not read the
frozen holdout.

After explicitly approving Fraunhofer model use, run the same five requests
through BGE-M3 retrieval and the Qwen-backed baseline and complete hybrid:

```bash
./scripts/run_workshop3_demo.sh --live
```

Live mode first prepares or reuses the BGE-M3 cache and Chroma index. It then
adds current baseline and hybrid dispositions, answers, triggers, and citations
to the HTML. The script uses only the configured Fraunhofer endpoint; it cannot
fall back to the official OpenAI Platform profile.

### Human-Calibrated LLM Judge

Prepare two blinded, family-disjoint 200-output annotation sets from a common
calibration result file:

```bash
uv run guardrails-llm prepare-judge-study \
  --source-results path/to/common_split_results.json \
  --output-dir path/to/human_judge_study
```

Preparation now prioritizes unique source requests before selecting a second
distinct system output for the same request. Exact duplicate review tasks are
excluded. The study-quality audit also checks German template errors, unique
question coverage, inclusion of complete in-house hybrid outputs, and usable
supporting evidence.

Run a separate local UI process for each reviewer. Use different ports when
both processes run on the same machine:

```bash
uv run guardrails-llm review-judge-study \
  --study-dir path/to/human_judge_study \
  --reviewer reviewer_a \
  --port 8765 \
  --open

uv run guardrails-llm review-judge-study \
  --study-dir path/to/human_judge_study \
  --reviewer reviewer_b \
  --port 8766 \
  --open
```

Each process binds only to `127.0.0.1` and exposes only the selected reviewer's
drafts. Field changes are transactionally autosaved in
`.judge_review.sqlite3`. Questions are grouped into collapsible sections, and
distinct anonymous system outputs remain separate judgments. When every item
in a section is either fully labeled or flagged as a dataset issue, the section
is atomically exported to the existing reviewer JSONL file. Dataset issues are
written to `judge_reviewer_a_issues.jsonl` or
`judge_reviewer_b_issues.jsonl`; they do not force reviewers to invent labels
for an unjudgeable item.

Do not share `judge_study_mapping.jsonl`, prediction files, the SQLite store,
or one reviewer's JSONL files with the other reviewer. Only the two blinded
item files and the reviewer's assigned UI process are needed.

Human rationales are optional; all five boolean dimensions remain required.
To create separate deterministic rubric recommendations without changing
either human file:

```bash
uv run guardrails-llm prepare-judge-recommendations \
  --study-dir path/to/human_judge_study
```

The UI hides recommendations by default and can reveal or copy one item, one
section, or the complete study. Every reveal and copy is written to the local
SQLite assistance log. Any affected item is an assisted review and must not be
reported as independently labeled. A trusted single operator may enable the
reviewer picker with `--allow-reviewer-switch`; do not use that flag when two
people are performing blinded independent review.

Check reviewer progress and reconcile two completed reviews locally:

```bash
uv run guardrails-llm judge-study-status \
  --study-dir path/to/human_judge_study
uv run guardrails-llm reconcile-judge-study \
  --study-dir path/to/human_judge_study
uv run guardrails-llm finalize-judge-study \
  --study-dir path/to/human_judge_study
```

After both human reviews and the recommendation files are complete, inspect all
three judgments side by side and adjudicate only Reviewer A versus Reviewer B
disagreements:

```bash
uv run guardrails-llm review-judge-reconciliation \
  --study-dir path/to/human_judge_study \
  --open
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
Predictions captured for an older study selection are not reusable after study
items are regenerated; recapture both judge candidates against the final item
IDs before calculating agreement.

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
`reports/final_calibration_evidence.md`. Existing checked-in output reflects
the historical pre-repair capture and must be rebuilt after the repaired-source
model runs. The command rejects holdout-derived or
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
