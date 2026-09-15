# LLM RAG Knowledge Guardrails

[![CI](https://github.com/zhittsova/llm-rag-knowledge-guardrails/actions/workflows/ci.yml/badge.svg)](https://github.com/zhittsova/llm-rag-knowledge-guardrails/actions/workflows/ci.yml)

A knowledge assistant needs to know when to answer and when to stop. This Python
project makes those decisions inspectable: **answer, block, redirect, or abstain**,
with configurable policy, permitted document scope, evidence checks, and evaluation.

I built it around a learning-assistant corpus. The architecture offers a starting
point for internal knowledge and support workflows; those business adaptations
have not been evaluated or deployed here.

By [Katerina Zhittsova](https://zhittsova.com). Read more about my work in
[AI Engineering](https://zhittsova.com/blog/).

## Try it locally

```bash
uv --directory guardrails-llm-deployment sync --frozen --dev
./scripts/run_guardrails_demo.sh --open
```

The comparison tour reads the current calibration reports. It shows the guardrail
funnel, four dispositions, LLM-judge validation, failure analysis, and the holdout
protocol. It makes no model API calls and exposes no holdout cases.

```bash
uv --directory guardrails-llm-deployment run guardrails-llm query \
  --profile local --retriever lexical --mode guardrailed \
  --question "What is retrieval augmented generation?"
```

For local policy editing, human review, and adjudication, follow the
[interface walkthrough](guardrails-llm-deployment/docs/interface_walkthrough.md).

## Choose a runtime

- `local`: lexical or hashing-based retrieval and extractive answers, without credentials.
- `openai-compatible`: your endpoint and explicit model names, using the Python `openai`
  package for embeddings, classification, answers, and verification. See the
  [configuration guide](guardrails-llm-deployment/docs/openai_compatible.md).
- `inhouse`: the fixed experiment profile used in the recorded results. Its model
  versions, calibrated settings, and endpoint restriction remain reproducible.

Remote commands require `--allow-remote-models`. A different backend needs its own
calibration; API compatibility alone does not transfer the results below.

## What the guardrails address

| Failure mode | Relevant controls | Possible outcome |
|---|---|---|
| Direct or obfuscated prompt injection | Normalization, regex/fuzzy rules, semantic intent checks, classifier | Block |
| Instructions hidden in retrieved text | Context sanitization and output inspection | Remove unsafe context, block, or abstain |
| Private-data requests or out-of-scope documents | Input guards and configured course/visibility/source filters | Block or exclude the source |
| Harmful requests and domain-policy violations | Input rules and domain classifier | Block or redirect |
| Unsupported answers or misleading citations | Evidence sufficiency, entailment, and citation checks | Abstain or retain supported claims |
| Sensitive content in generated output | Output PII and injection rules | Block |

These controls address observed failure modes; they are not a guarantee against all
attacks. Document filters require correct metadata. An authenticated multi-user
permission system is outside the current prototype.

## Runtime flow

```text
request -> input checks -> semantic intent -> model classification
        -> filtered retrieval -> evidence gate -> candidate answer
        -> entailment and citation checks -> output inspection -> release
```

Requests can stop early. Policy redirects use a separate tutoring response path.
The offline judge evaluates saved outputs; it is separate from the live guardrails.

## Repository layout

```text
guardrails-llm-deployment/
  src/guardrails_llm/   runtime, retrieval, guardrails, models, and evaluation
  data/                 corpora, policies, and versioned evaluation cases
  docs/                 technical workflows and corpus contract
  reports/              compact, versioned evaluation evidence
  tests/                deterministic pytest coverage
scripts/                repository-level demo entry points
A_Configurable_Hybrid_Guardrail_Architecture_for_a_Retrieval_Augmented_AI_Learning_Assistant/
                        IEEE report source and bibliography
output/pdf/             compiled report
Workshop1..3/           workshop planning and presentation material
```

## In-house profile

The `inhouse` profile is restricted to the configured Fraunhofer endpoint. It
uses `BAAI/bge-m3` for embeddings and `Qwen/Qwen3.6-35B-A3B` for
classification, answer generation, and entailment verification.

From `guardrails-llm-deployment/`, inspect the credential-safe configuration
before allowing a remote call:

```bash
uv run guardrails-llm model-config --profile inhouse
uv run guardrails-llm validate-runtime-config
```

`data/guardrail_runtime_inhouse.toml` records the model names, thresholds,
retrieval depths, and artifact paths. Capture manifests include its SHA-256
hash. Credentials, generated indexes, and caches remain local and must not be
committed.

The local instructor policy manager edits and validates the TOML guardrail
policy without making remote calls:

```bash
uv --directory guardrails-llm-deployment run guardrails-llm manage-policy --open
```

## Results and evidence boundary

| Evaluation | Result | Scope |
|---|---|---|
| Complete in-house hybrid | 392/400 correct, macro-F1 0.980, no unsafe answers | Common calibration split |
| Qwen domain classifier | 597/600 overall, 148/150 on calibration | Balanced component benchmark |
| BGE-M3 retrieval | 0.925 expected-document recall@3 | 200 evidence-bearing calibration cases |
| Qwen3Guard strict policy | 400/400, compared with 398/400 for Qwen | Shared binary safety component task |
| MiniMax judge validation | 0.795 exact five-dimension agreement | Family-disjoint 200-output validation set |

These are calibration and component results, not frozen-holdout results. The
Qwen3Guard score applies only to the strict binary safety mapping. One human
annotator completed two recommendation-assisted passes for the judge study and
reconciled the three discrepancies. These labels provide an assisted human
reference, not inter-annotator evidence.

Expected-document citation precision is `0.766`, below the additional `0.95`
diagnostic gate. The runtime verifier accepted every emitted citation, but the
expected-document labels may omit valid sources. Independent human review is
needed before the difference can be attributed to a citation defect.

The frozen 400-case holdout remains unopened. A final holdout run requires two
independent reviews, adjudication, judge validation, and a sealed runtime
configuration. The repository provides fail-closed commands for each gate.

Detailed evidence is available in:

- [`final_calibration_evidence.md`](guardrails-llm-deployment/reports/final_calibration_evidence.md)
- [`inhouse_judge_validation_v23.md`](guardrails-llm-deployment/reports/inhouse_judge_validation_v23.md)
- [`guardrails-llm-deployment/README.md`](guardrails-llm-deployment/README.md)

## Development

The Python package is `guardrails_llm`, and the installed command is
`guardrails-llm`. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a
pull request. The package README documents the complete preparation, capture,
review, and evaluation workflows.
