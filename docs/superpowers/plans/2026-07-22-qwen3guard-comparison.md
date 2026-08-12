# Qwen3Guard Comparison Implementation Record

## Status

Completed. The branch adds a native Qwen3Guard adapter, resumable capture,
provenance-checked offline comparison, CLI commands, tests, documentation, and
the derived matched-comparison evidence.

## Implemented components

- `qwen3guard.py` defines the native result contract, parser, category mapping,
  provider adapter, and explicit model-unavailable error.
- `qwen3guard_experiment.py` selects the balanced 600-case benchmark, captures
  native responses, writes resumable manifests, and produces the offline
  comparison.
- The CLI exposes `capture-qwen3guard-classifier`,
  `evaluate-qwen3guard-classifier`, and `compare-qwen3guard-classifier`.
- The README and contributor guide document remote-call safety, evidence
  provenance, interpretation, and reproduction.

## Completed evidence run

The matched run contains 600 valid responses from each model. The primary
binary task uses 400 taxonomy-compatible cases. The other 200 cases are kept
as a descriptive taxonomy-gap analysis.

- Qwen3Guard strict policy: 400/400 correct.
- Qwen3Guard permissive policy: 331/400 correct.
- Prompted Qwen3 baseline: 398/400 correct.
- Frozen holdout cases used: 0.

The saved result is tied to the dataset revision at commit `2db138b`. Its
dataset-manifest, split, selection, capture, and model identifiers are recorded
in the artifacts. The evaluator reproduces the saved JSON exactly when given
that revision and the original captures, and it rejects the current revised
splits because their hashes differ.

## Verification completed

- Native-output parsing and category mapping tests pass.
- Capture resume and provider-unavailability tests pass.
- Incomplete and mismatched capture manifests are rejected.
- Strict and permissive release policies are tested independently.
- Historical capture field names are supported.
- No test makes a remote model call.
- The full repository test suite passes after synchronization with `main`.
- Changed public files contain no credentials or Cyrillic text.
- The feature diff contains no report source, PDF, frozen-holdout output, or raw
  JSONL capture.

## Versioned artifacts

- `guardrails-llm-deployment/reports/qwen3guard_gen_4b_600_matched_v1_manifest.json`
- `guardrails-llm-deployment/reports/qwen3guard_vs_qwen3_600_matched_v1_evaluation.json`

The raw model captures remain outside version control. This keeps the PR
focused while retaining enough hashed provenance to validate the derived
report against the original local evidence.
