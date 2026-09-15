# In-House LLM Judge Validation

MiniMax `MiniMaxAI/MiniMax-M2.5` with rubric
`guardrail-judge-v2.3` was selected on 200 judge-calibration outputs and then
evaluated without further tuning on 200 family-disjoint judge-validation
outputs. All 400 responses were valid structured JSON.

| Metric | Calibration | Validation | Required |
|---|---:|---:|---:|
| Structured validity | 1.000 | 1.000 | 1.000 |
| Exact five-dimension agreement | 0.790 | 0.795 | >= 0.750 |
| Groundedness agreement | 0.920 | 0.915 | >= 0.850 |
| Lowest dimension agreement | 0.880 | 0.900 | >= 0.800 |

Validation dimension agreement was `0.915` groundedness, `0.995` privacy,
`0.995` injection safety, `0.900` academic integrity, and `0.970` refusal
appropriateness. Every judge-validation quality gate passed.

One human annotator labelled all 400 outputs in two separate passes. The passes
agreed on all five dimensions for 397 items, after which the same annotator
reviewed and reconciled the three discrepancies. GPT-5.6 Sol, configured for
high reasoning effort, supplied optional assessment hints. The local interface
kept these hints hidden by default and logged 67 assistance actions during the
first pass and 40 during the second. The resulting labels provide an assisted
human reference and evidence of within-annotator consistency, not
inter-annotator agreement. The LLM judge is used only as a secondary evaluator.

This result validates the judge workflow. It is not frozen-holdout system
performance. The holdout remains unopened until its separate double review,
adjudication, dataset sealing, and runtime configuration freeze are complete.
