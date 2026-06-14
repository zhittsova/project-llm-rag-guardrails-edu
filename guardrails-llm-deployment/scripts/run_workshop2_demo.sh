#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

CORPUS="${CORPUS:-data/python_course_docs.jsonl}"
COURSE_ID="${COURSE_ID:-python-intro}"
INDEX_DIR="${INDEX_DIR:-indexes/python-course-chroma}"
REPORT="${REPORT:-reports/python_course_rag_demo.html}"
QUESTION="${QUESTION:-What is declarative knowledge?}"

if command -v uv >/dev/null 2>&1; then
  UV_BIN="uv"
elif [ -x /opt/homebrew/bin/uv ]; then
  UV_BIN="/opt/homebrew/bin/uv"
elif [ -x /usr/local/bin/uv ]; then
  UV_BIN="/usr/local/bin/uv"
elif [ -x "${HOME}/.cargo/bin/uv" ]; then
  UV_BIN="${HOME}/.cargo/bin/uv"
elif [ -x "${HOME}/.local/bin/uv" ]; then
  UV_BIN="${HOME}/.local/bin/uv"
else
  echo "uv is required but was not found in PATH or common install locations." >&2
  exit 127
fi

case "${INDEX_DIR}" in
  indexes/*) ;;
  *)
    echo "Refusing to clear INDEX_DIR outside indexes/: ${INDEX_DIR}" >&2
    exit 2
    ;;
esac

echo "== Workshop 2 RAG demo =="
echo "Corpus: ${CORPUS}"
echo "Course ID: ${COURSE_ID}"
echo "Question: ${QUESTION}"
echo

echo "== 1. Validate corpus =="
"${UV_BIN}" run guardrails-llm validate-corpus --corpus "${CORPUS}"
echo

echo "== 2. Rebuild vector index =="
rm -rf "${INDEX_DIR}"
"${UV_BIN}" run guardrails-llm build-index \
  --corpus "${CORPUS}" \
  --index-dir "${INDEX_DIR}"
echo

echo "== 3. Run guardrailed vector query =="
"${UV_BIN}" run guardrails-llm query \
  --mode guardrailed \
  --course-id "${COURSE_ID}" \
  --retriever vector \
  --corpus "${CORPUS}" \
  --index-dir "${INDEX_DIR}" \
  --question "${QUESTION}"
echo

echo "== 4. Write HTML visualization =="
"${UV_BIN}" run guardrails-llm visualize \
  --corpus "${CORPUS}" \
  --course-id "${COURSE_ID}" \
  --retriever vector \
  --index-dir "${INDEX_DIR}" \
  --mode guardrailed \
  --question "${QUESTION}" \
  --output "${REPORT}"
echo

echo "HTML report: ${PROJECT_DIR}/${REPORT}"
