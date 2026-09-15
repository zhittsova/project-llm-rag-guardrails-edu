#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
exec uv --directory guardrails-llm-deployment run guardrails-llm guardrails-demo \
  --judge-evidence reports/inhouse_judge_validation_v23.json \
  --output ../output/demo/guardrails.html "$@"
