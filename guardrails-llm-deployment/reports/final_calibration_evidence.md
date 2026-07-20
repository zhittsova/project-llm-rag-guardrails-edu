# Final Calibration Evidence

Calibration evidence only: the frozen holdout remains unopened.

## Common-Split Technique Comparison

| Technique | Correct | Accuracy | Macro-F1 | False refusal | Unsafe answered |
|---|---:|---:|---:|---:|---:|
| Baseline RAG | 100/400 | 0.250 | 0.100 | 0.000 | 1.000 |
| Normalized regex + metadata | 141/400 | 0.352 | 0.271 | 0.025 | 0.790 |
| Fuzzy + shared controls | 153/400 | 0.383 | 0.333 | 0.025 | 0.855 |
| BGE similarity + shared controls | 248/400 | 0.623 | 0.583 | 0.045 | 0.640 |
| Deterministic hybrid policy | 249/400 | 0.625 | 0.584 | 0.045 | 0.635 |
| Qwen classifier scenario | 389/400 | 0.973 | 0.972 | 0.055 | 0.000 |
| Complete in-house hybrid | 392/400 | 0.980 | 0.980 | 0.040 | 0.000 |

## Accuracy Confidence Intervals

| Technique | Row-level accuracy 95% CI | Family-level accuracy 95% CI |
|---|---:|---:|
| Baseline RAG | [0.2075, 0.2900] | [0.1107, 0.5152] |
| Normalized regex + metadata | [0.3050, 0.3975] | [0.1678, 0.6312] |
| Fuzzy + shared controls | [0.3375, 0.4300] | [0.2294, 0.5935] |
| BGE similarity + shared controls | [0.5725, 0.6700] | [0.3431, 0.8339] |
| Deterministic hybrid policy | [0.5750, 0.6725] | [0.3431, 0.8386] |
| Qwen classifier scenario | [0.9550, 0.9875] | [0.9614, 0.9900] |
| Complete in-house hybrid | [0.9650, 0.9925] | [0.9667, 0.9964] |

## Complete Hybrid

- Correct behavior: 392/400.
- Macro-F1: 0.980.
- Answer recall: 0.920.
- Block, abstain, and redirect recall: 1.000, 1.000, and 1.000.
- Supported-answer precision: 1.000.
- Citation-entailment precision (model-verifier-conditioned): 1.000 across 303 citations.
- Expected-document citation precision: 0.766.

The expected-document citation diagnostic remains failed and visible. Generated expected-document labels require independent human review before this metric can be treated as authoritative.

The citation-entailment figure is runtime verifier consistency, not an independent human entailment judgment.

## Evidence Boundary

Human judge agreement, independently adjudicated source labels, and the final 400-case frozen-holdout run are not yet available. Calibration results guide engineering decisions but are not a final generalization claim.

## Remaining Complete-Hybrid Failures

All 8 behavior errors are false abstentions:

- 2 retrieval misses.
- 2 answerability abstentions with the expected document present.
- 4 entailment rejects caused by unsupported extra claims.
