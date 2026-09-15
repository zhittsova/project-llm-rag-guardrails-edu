# OpenAI-compatible model endpoints

The `openai-compatible` profile uses the Python `openai` client with your endpoint
and model names. It does not select the recorded experiment's provider or models.
The current adapter requires Chat Completions with `response_format=json_object`
for structured model decisions, and the embeddings endpoint for vector retrieval.
Test those capabilities before treating a backend as supported.

Keep credentials in a local, ignored `.env` file or your shell environment. Set
`OPENAI_BASE_URL` to your API base URL, including its API prefix, and
`OPENAI_API_KEY` to the backend credential. `OPENAI_API_URL` is a supported alias
when `OPENAI_BASE_URL` is unset. Never commit credentials.

Inspect the host and credential-presence flag without calling a model:

```bash
uv --directory guardrails-llm-deployment run guardrails-llm model-config \
  --profile openai-compatible
```

Choose an embedding model supported by your endpoint and build a separate index.
Replace every model placeholder below with an actual model identifier:

```bash
uv --directory guardrails-llm-deployment run guardrails-llm build-index \
  --profile openai-compatible \
  --embedding-model YOUR_EMBEDDING_MODEL \
  --index-dir indexes/my-backend \
  --allow-remote-models

uv --directory guardrails-llm-deployment run guardrails-llm query \
  --profile openai-compatible --mode guardrailed \
  --embedding-model YOUR_EMBEDDING_MODEL \
  --answer-model YOUR_ANSWER_MODEL \
  --classifier-model YOUR_CLASSIFIER_MODEL \
  --entailment-model YOUR_VERIFIER_MODEL \
  --index-dir indexes/my-backend \
  --question "What is retrieval augmented generation?" \
  --allow-remote-models
```

The corpus, course ID, policy, and index path stay under your control. Index and query
must use the same embedding model. Use a separate index and cache for each embedding
configuration. With no custom policy, the default policy is used; remote semantic
policy rules require an explicitly selected policy and appropriate calibration.

This profile enables the configured remote roles but never enables remote-call
permission automatically. Thresholds are not calibrated for your backend. Start
with development cases, then evaluate safety and useful-answer recall together.
Keep the frozen holdout out of that tuning loop.

The existing `inhouse` profile retains the original experiment settings and endpoint
check. Historical result files keep their exact model names for reproducibility.
