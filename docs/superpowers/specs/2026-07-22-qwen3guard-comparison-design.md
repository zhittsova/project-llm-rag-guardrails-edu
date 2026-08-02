# Qwen3Guard Classifier Comparison Design

## Objective

Add `qwen3guard-gen-4b` as a specialized guardrail-classifier candidate and
compare it with the existing prompted `Qwen/Qwen3.6-35B-A3B` classifier on the
same versioned 600-case component benchmark.

The integration must preserve Qwen3Guard's native moderation contract instead
of prompting it to imitate the project's six-label classifier. No remote model
call may run without `--allow-remote-models`.

## Why a dedicated adapter is required

The existing Qwen classifier returns one project label:

```text
safe
prompt_injection
pii
academic_integrity
unsafe_request
unsupported
```

Qwen3Guard-Gen natively returns a severity (`Safe`, `Controversial`, or
`Unsafe`) and one or more safety categories. Its taxonomy does not directly
represent RAG evidence insufficiency, and it may not distinguish educational
integrity from broader unsafe or unethical intent.

Forcing Qwen3Guard into the six-label prompt would test generic instruction
following rather than its intended moderation behavior. The comparison will
therefore retain the native output and report taxonomy limitations explicitly.

## Architecture

### Native output parser

Add a focused parser for Qwen3Guard prompt-moderation responses. It will:

- accept case-insensitive `Safety:` and `Categories:` fields;
- normalize the severity to `safe`, `controversial`, or `unsafe`;
- normalize categories while preserving the original model text for audit;
- reject missing or unknown severity values as malformed responses;
- never infer a successful decision from malformed output.

### Model adapter

Add an OpenAI-compatible Qwen3Guard classifier adapter that sends the user
request as a user message to `qwen3guard-gen-4b`. The adapter will use the
existing endpoint configuration, remote-call gate, retry behavior, and capture
manifest conventions.

The adapter will expose both:

- the native moderation result;
- a conservative project comparison result derived through an explicit,
  versioned category mapping.

### Comparison levels

The same 600 benchmark cases will be evaluated at two levels.

#### Level 1: intervention detection

The primary cross-model comparison asks whether a request should continue
without a guardrail intervention.

- expected `safe` -> allow;
- all other project labels -> intervene.

Qwen3Guard `Safe` maps to allow. `Controversial` and `Unsafe` map to intervene.
This measures broad guardrail detection without pretending the specialized
model can choose the correct RAG action.

#### Level 2: project-label coverage

Where a native category maps unambiguously, report the corresponding project
label. Ambiguous or unsupported mappings remain `unmapped`; they are not
silently counted as correct.

The report will show:

- exact project-label accuracy on mapped cases;
- mapping coverage over all 600 cases;
- per-project-label mapped, correct, incorrect, and unmapped counts;
- a note that `unsupported` is an evidence-policy category outside native
  safety moderation;
- any ambiguity involving academic-integrity cases.

## Capture and report artifacts

The capture command will be resumable and will write JSONL predictions plus a
manifest containing:

- model and endpoint host;
- parser and mapping versions;
- prompt/input hashes;
- case count and completion status;
- malformed responses and provider errors;
- no API key and no stored credential.

The comparison report will place the existing Qwen six-label result beside the
Qwen3Guard native result. It will report intervention accuracy, precision,
recall, F1, safe false-positive rate, unsafe false-negative rate, structured
response validity, project-label mapping coverage, and per-label limitations.

## CLI behavior

Add explicit commands or options for:

1. capturing Qwen3Guard predictions for the 600-case benchmark;
2. evaluating the capture;
3. producing a comparison with an existing Qwen classifier evaluation.

All commands that can contact the endpoint require
`--allow-remote-models`. Offline evaluation of existing captures does not.

If the provider reports that the model is unavailable, the capture must stop
with a clear actionable error and preserve already completed rows. The report
must never substitute simulated results for a real model run.

## Testing

Implementation follows test-driven development.

Tests will cover:

- valid `Safe`, `Controversial`, and `Unsafe` output parsing;
- multiple and absent categories;
- malformed and unknown severity output;
- remote-call refusal without explicit allowance;
- conservative intervention mapping;
- explicit unmapped project categories;
- 600-case report accounting invariants;
- resume behavior and manifest contents;
- credential exclusion;
- CLI argument wiring with fake model clients.

No test will call the Fraunhofer API.

## Documentation

Update the package README and contributor documentation with:

- the purpose of the Qwen3Guard comparison;
- the difference between native safety moderation and six-label RAG policy
  classification;
- commands for capture and offline evaluation;
- the current provider-availability limitation;
- the rule that results may only be claimed after a completed real capture.

## Out of scope

- using Qwen3Guard as an LLM-as-a-judge;
- replacing the Qwen answer generator or entailment verifier;
- using Qwen3Guard-Stream;
- official OpenAI Platform execution;
- changing the frozen holdout;
- presenting placeholder or simulated scores.
