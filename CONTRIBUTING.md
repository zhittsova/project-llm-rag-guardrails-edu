# Contributing

## Setup

```bash
cd guardrails-llm-deployment
uv sync --dev
uv run pytest
```

Python 3.11 through 3.14 is exercised in CI.

## Branch and Commit Workflow

- Start each workstream from updated `main` on a dedicated branch.
- Use branch names such as `feat/short-description` or
  `fix/short-description`.
- Keep commits small and use conventional one-line titles no longer than 80
  characters.
- Do not add co-authored trailers.
- Open a PR, wait for the complete CI matrix, address review findings, and use
  squash merge.
- Keep local `main` fast-forward-only after merges.

## Required Checks

Run focused tests while developing, then before a PR run:

```bash
cd guardrails-llm-deployment
uv run pytest
uv build
git diff --check
```

Add tests for changed behavior. Model adapters should use fake clients in tests;
normal test runs must not call a remote API.

## Remote Models and Secrets

- Never commit `.env`, API keys, raw credentials, generated vector indexes, or
  embedding caches.
- Keep all remote calls behind `--allow-remote-models`.
- Use only the provider selected for the current workstream.
- Manifests may record endpoint host, key presence, model names, prompt
  versions, thresholds, and hashes, but never secret values.
- Retain provider and parsing failures in evaluation artifacts instead of
  silently converting them into successful blocks.

## Evaluation Discipline

- Tune rules, prompts, and thresholds on development and calibration only.
- Do not run or inspect the frozen holdout until its review gate is complete.
- Keep paraphrase siblings in the same split.
- Report per-class failures and failed gates; do not hide them behind an
  aggregate score.
- Human labels remain ground truth for judge calibration.
- Keep reviewer drafts isolated. A reviewer UI must not expose technique
  mappings, model predictions, hidden labels, or another reviewer's data.
- Treat recommendation-assisted labels as assisted review, not independent
  human annotation. Preserve reveal and copy provenance.
- Keep demos fail-closed: offline mode must not contact a provider, and live
  mode must retain an explicit remote-model gate and evidence-scope labels.
- Keep write-capable local administration tools bound to localhost, reject
  non-local browser origins, validate drafts before publish, write atomically,
  and retain rollback snapshots.
- Treat malformed or ambiguous study items as dataset issues instead of
  forcing a label. Repair and regenerate the study before final reconciliation.
- Checked-in reports should be compact, credential-safe, versioned, and clear
  about whether they are development, calibration, or holdout evidence.
- Keep large resumable prediction and per-case result captures local. Commit
  their configuration manifests and compact derived reports instead.
- Compare classifier models only on identical complete case IDs and preserve
  the capture manifest for each model. Reject partial or misaligned comparisons.
- Report native taxonomy coverage separately from intervention behavior. Do not
  force project labels into unsupported provider categories.
- Never publish simulated model scores. Provider unavailability, malformed
  responses, missing rows, and failed quality gates must remain visible.
- Update the package README whenever a model comparison adds a command, model,
  mapping rule, evidence artifact, or interpretation limitation.

## Documentation Gate for Every PR

Every PR must review:

1. the repository-level `README.md`;
2. `guardrails-llm-deployment/README.md`;
3. this contributor guide;
4. any technical workflow directly changed by the PR.

Update them when setup, commands, architecture, evaluation evidence,
limitations, or contribution workflow changes. If no edit is needed, record
`reviewed; no change needed` in the workstream progress notes.

## Pull Request Checklist

- [ ] The change is scoped to one workstream.
- [ ] Focused tests and the full test suite pass.
- [ ] The package builds.
- [ ] No credentials or local artifacts are included.
- [ ] Remote model behavior remains explicitly gated.
- [ ] Evaluation split and provenance rules are preserved.
- [ ] Failed quality gates and limitations remain visible.
- [ ] Both READMEs and contributor documentation were reviewed.
