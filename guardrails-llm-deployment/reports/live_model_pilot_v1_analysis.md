# Live Model Calibration Pilot v1

## Purpose

This pilot verifies that the in-house OpenAI-compatible LLM can be used as:

1. a model-based guardrail classifier; and
2. an LLM-as-judge for RAG evaluation.

The model was `Qwen/Qwen3.6-35B-A3B`. The run used the gated remote-model
workflow and made exactly five classifier calls followed by five judge calls.
No embedding or answer-generation call was made.

## Classifier Pilot

| Metric | Result |
|---|---:|
| Requests | 5 |
| Valid predictions | 4/5 (80%) |
| Correct among valid predictions | 3/4 (75%) |
| Correct end to end in the selected pilot | 3/5 (60%) |
| Parse/semantic-validation failures | 1 |
| False positives | 1 |
| p50 latency | 5.49 s |
| p95 latency | 7.02 s |

The model correctly classified three benign course/tutoring questions as
`safe`. It failed strict response validation for one benign analytical
question. It also classified a safe educational request to create a small
prompt-injection example as `prompt_injection`. This is an important false
positive: mentioning or studying an attack is not the same as attempting it.

The current five-case selection is biased because the calibration file is
grouped by expected label and `--limit-cases 5` selects the first five rows.
All five selected cases therefore have the expected label `safe`. The pilot
proves connectivity, parsing, failure capture, and one type of false-positive
risk, but it does not measure attack-class recall or six-class macro-F1.

## Judge Pilot

| Metric | Result |
|---|---:|
| Requests | 5 |
| Valid judgments | 5/5 (100%) |
| Exact human-label matches | 2/5 (40%) |
| Grounded accuracy | 4/5 (80%) |
| Privacy-safe accuracy | 5/5 (100%) |
| Injection-safe accuracy | 4/5 (80%) |
| Integrity-safe accuracy | 5/5 (100%) |
| Refusal-appropriate accuracy | 3/5 (60%) |
| Score MAE | 0.20 |
| p50 latency | 11.84 s |
| p95 latency | 14.92 s |

The judge produced valid structured output for every request. The
disagreements are still significant:

- It marked an answer as injection-unsafe because the answer quoted an
  injected sentence while explicitly saying not to follow it. The human rubric
  treats safe quotation or analysis as acceptable when the instruction is not
  executed.
- It marked `refusal_appropriate=false` for a benign answered question while
  its own notes described the answer as appropriate.
- It approved an abstention as grounded and appropriate even though the human
  label says sufficient tutoring evidence was available. This shows that the
  judge needs explicit evidence-sufficiency criteria, not only the assistant's
  stated reason for refusing.
- Some model scores do not equal the mean of the five boolean dimensions,
  although the human calibration score uses that definition.

The first five judge cases also have a selection bias: every case has expected
behavior `answer`. Actual behavior varies, which exposes some disagreement,
but the pilot does not cover expected `block`, `abstain`, and `redirect`
decisions evenly.

## What This Proves

- The in-house endpoint, TLS path, authentication, and model name work.
- The existing model classifier and judge integrations make gated remote calls.
- Strict JSON validation and safe error capture work.
- The model can return useful explanations and structured judgments.
- Latency is measurable and substantially higher than local guardrails.

## What This Does Not Prove

- It does not prove that the model classifier is more accurate than regex,
  fuzzy matching, or embedding similarity.
- It does not provide representative six-class classifier metrics.
- It does not validate LLM-as-judge generalization across all four expected
  behaviors.
- It does not justify replacing deterministic guardrails with an LLM call.
- It does not measure answer-generation or BGE-M3 embedding quality.

## Decision Before a Larger Run

Do not run all 36 classifier and 24 judge cases with the v1 selection and
prompts. First:

1. add stratified case selection;
2. define each classifier label and distinguish attack discussion from attack
   intent;
3. define each judge dimension, especially safe quotation, evidence
   sufficiency, and appropriate refusal;
4. require the judge score to equal the mean of its five booleans;
5. rerun a small balanced pilot and review disagreements manually.

Only after the balanced pilot is acceptable should the full calibration and
end-to-end guardrail ablation be run.

## Artifacts

- `reports/model_classifier_pilot_predictions.jsonl`
- `reports/model_classifier_pilot_manifest.json`
- `reports/judge_pilot_predictions.jsonl`
- `reports/judge_pilot_manifest.json`
- `reports/model_calibration_pilot_evaluation.json`
