#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_DIR/.uv-cache}"

run_demo() {
  local args=(workshop3-demo)
  if [[ "${1:-}" == "live" ]]; then
    args+=(--live --allow-remote-models)
  fi
  if [[ -n "${ENV_FILE:-}" ]]; then
    args+=(--env-file "$ENV_FILE")
  fi
  if [[ "${OPEN_BROWSER:-1}" == "1" ]]; then
    args+=(--open)
  fi
  uv run guardrails-llm "${args[@]}"
}

if [[ "${1:-}" == "--live" ]]; then
  prepare_args=(prepare-inhouse-bge --allow-remote-models)
  if [[ -n "${ENV_FILE:-}" ]]; then
    prepare_args+=(--env-file "$ENV_FILE")
  fi
  uv run guardrails-llm "${prepare_args[@]}"
  run_demo live
else
  run_demo offline
fi
