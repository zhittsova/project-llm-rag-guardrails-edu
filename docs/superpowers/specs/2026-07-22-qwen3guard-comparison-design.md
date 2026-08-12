# Qwen3Guard Classifier Comparison Design

## Objective

This work evaluates `qwen3guard-gen-4b` as a specialized safety classifier
alongside the existing prompted `Qwen/Qwen3.6-35B-A3B` classifier. It does not
replace the answer generator, retrieval checks, entailment verifier, or judge.

The comparison preserves Qwen3Guard's native moderation output instead of
prompting it to imitate the project's six-label classifier. Remote capture is
available only behind the existing `--allow-remote-models` gate.

## Native adapter

The adapter sends the request as a user message and parses the native fields:

```text
Safety: Safe | Controversial | Unsafe
Categories: <zero or more native categories>
```

Severity and categories are normalized for evaluation, while the raw model
text remains in the capture for audit. Missing or unknown severity values are
recorded as failed predictions. A provider response showing that the model is
unavailable stops the run and preserves completed rows and the partial
manifest.

## Evaluation scope

The balanced component benchmark contains 600 development and calibration
cases, with 100 cases for each project label. It does not use the frozen
holdout.

The primary binary comparison contains the 400 cases whose labels have a
shared safety interpretation:

- `safe` maps to safe;
- `prompt_injection`, `pii`, and `unsafe_request` map to unsafe.

Two Qwen3Guard release policies are reported separately. The strict policy
maps both `Controversial` and `Unsafe` to unsafe. The permissive policy maps
`Controversial` to safe. The Qwen baseline maps its predicted `safe` label to
safe and every predicted risk label to unsafe.

The remaining 200 `academic_integrity` and `unsupported` cases are not included
in the primary score. They form a descriptive taxonomy-gap analysis because
Qwen3Guard does not natively represent course evidence sufficiency, and its
safety categories do not provide an equivalent academic-integrity policy.

## Evidence and provenance

Capture manifests identify the model, dataset version, dataset-manifest hash,
split hashes, selected-case hash, case count, completion status, and request
policy. The offline comparison rejects either capture when the manifests are
incomplete, contain failed rows, or do not match the supplied dataset and case
selection.

The versioned report includes:

- binary accuracy, macro F1, unsafe recall, and safe false-positive rate;
- confusion matrices and per-class metrics;
- strict and permissive Qwen3Guard results;
- native severity and category distributions;
- the descriptive taxonomy-gap counts;
- latency summaries and hashes for every comparison input.

Raw captures are retained outside version control because they contain the
complete model-output trace. The repository stores the Qwen3Guard manifest and
the derived comparison JSON needed to audit the published claims.

## Interpretation limits

The saved capture uses the dataset revision at commit `2db138b`. The current
split files have different hashes, so the evaluator deliberately rejects them
for that capture. Reproduction must use the recorded revision and original
captures, or a new matched capture must be made on the current revision.

The results measure one classifier component, not the end-to-end RAG system.
The latency values come from sequential provider captures made at different
times, so they are descriptive rather than a controlled performance benchmark.

## Tests

Deterministic tests cover parsing, native-category mapping, remote-call gating,
resume behavior, provider unavailability, failed-row manifests, matched case
IDs, strict and permissive policies, historical field aliases, and provenance
rejection. Tests use fake clients and never call the Fraunhofer endpoint.
