# Final Calibration Evidence

Calibration evidence only: the frozen holdout remains unopened.

## Common-Split Technique Comparison

| Technique | Correct | Accuracy | Macro-F1 | False refusal | Unsafe answered |
|---|---:|---:|---:|---:|---:|
| Baseline RAG | 100/400 | 0.250 | 0.100 | — | 1.000 |
| Normalized regex + metadata | 203/400 | 0.507 | 0.420 | 0.255 | 0.255 |
| Fuzzy + shared controls | 215/400 | 0.537 | 0.485 | 0.180 | 0.285 |
| BGE similarity + shared controls | 316/400 | 0.790 | 0.788 | 0.085 | 0.165 |
| Deterministic hybrid policy | 317/400 | 0.792 | 0.790 | 0.080 | 0.165 |
| Qwen classifier scenario | 394/400 | 0.985 | 0.985 | 0.030 | 0.000 |
| Complete in-house hybrid | 391/400 | 0.978 | 0.978 | 0.045 | 0.000 |

## Accuracy Confidence Intervals

| Technique | Row-level accuracy 95% CI | Family-level accuracy 95% CI |
|---|---:|---:|
| Baseline RAG | [0.2075, 0.2900] | [0.1107, 0.5152] |
| Normalized regex + metadata | [0.4600, 0.5550] | [0.2244, 0.8177] |
| Fuzzy + shared controls | [0.4925, 0.5850] | [0.3166, 0.7848] |
| BGE similarity + shared controls | [0.7525, 0.8300] | [0.6054, 0.8959] |
| Deterministic hybrid policy | [0.7550, 0.8300] | [0.6054, 0.8998] |
| Qwen classifier scenario | [0.9725, 0.9950] | [0.9787, 0.9917] |
| Complete in-house hybrid | [0.9625, 0.9925] | [0.9631, 0.9942] |

## Complete Hybrid

- Correct behavior: 391/400.
- Macro-F1: 0.978.
- Answer recall: 0.910.
- Block, abstain, and redirect recall: 1.000, 1.000, and 1.000.
- Supported-answer precision: 1.000.
- Citation-entailment precision: 1.000.
- Expected-document citation precision: 0.752.

The expected-document citation diagnostic remains failed and visible. Generated expected-document labels require independent human review before this metric can be treated as authoritative.

## Evidence Boundary

Human judge agreement, independently adjudicated source labels, and the final 400-case frozen-holdout run are not yet available. Calibration results guide engineering decisions but are not a final generalization claim.

## Remaining Complete-Hybrid Failures

All 9 behavior errors are false abstentions:

- 2 retrieval misses.
- 3 evidence-gate rejects with the expected document present.
- 4 entailment rejects caused by unsupported extra claims.
