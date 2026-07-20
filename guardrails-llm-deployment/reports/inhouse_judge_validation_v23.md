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

Two reviewers labelled all 400 outputs and agreed on all five dimensions for
397 items; three disagreements were adjudicated. Recommendation assistance was
recorded for 67 Reviewer A actions and 40 Reviewer B actions. The labels are
therefore recommendation-assisted rather than fully independent double
annotation. Adjudicated human labels remain ground truth, and the LLM judge is
used only as a secondary evaluator.

This result validates the judge workflow. It is not frozen-holdout system
performance. The holdout remains unopened until its separate double review,
adjudication, dataset sealing, and runtime configuration freeze are complete.
