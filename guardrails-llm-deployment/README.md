# Guardrails in LLM Deployment

Prototype for the AMT project: develop and evaluate best practices for guardrails in an LLM-based learning assistant.

The first implementation is deliberately local and deterministic. It models the deployment pipeline from the Workshop 1 presentation without requiring an API key:

- baseline RAG assistant over a small course corpus
- guardrailed RAG assistant with input, retrieval, output, and logging controls
- JSONL evaluation set with normal and adversarial cases
- quantitative comparison between baseline and guardrailed runs

## Project Structure

```text
src/guardrails_llm/
  baseline_pipeline.py  baseline RAG without guardrails
  cli.py          command line interface
  corpus.py       document loading and chunking
  retrieval.py    lexical retrieval baseline
  course_corpus.py markdown course corpus normalization
  vector.py       local Chroma vector retrieval
  visualization.py static HTML RAG pipeline reports
  guards.py       input/output guardrails
  pipeline.py     guardrailed assistant and assistant factory
  evaluation.py   JSONL test runner and metrics
data/
  course_docs.jsonl
  eval_cases.jsonl
  python_course_docs.jsonl
docs/
  corpus_contract.md
  project_plan.md
  workshop2_demo.md
  workshop2_rag_slides.html
  workshop2_requirements.md
  workshop2_slides.html
tests/
  test_guards.py
  test_pipeline.py
```

## Quick Start

The Python import package uses underscores: `guardrails_llm`.
The installed console script uses hyphens: `guardrails-llm`.

From the repository root:

```bash
./scripts/run_workshop2_demo.sh
uv --directory guardrails-llm-deployment run guardrails-llm query --mode guardrailed --retriever langchain --question "What is retrieval augmented generation?"
uv --directory guardrails-llm-deployment run guardrails-llm validate-corpus --corpus data/course_docs.jsonl
uv --directory guardrails-llm-deployment run guardrails-llm build-index --corpus data/course_docs.jsonl --index-dir indexes/chroma
uv --directory guardrails-llm-deployment run guardrails-llm query --mode guardrailed --retriever vector --index-dir indexes/chroma --question "What is retrieval augmented generation?"
uv --directory guardrails-llm-deployment run guardrails-llm build-index --corpus data/python_course_docs.jsonl --index-dir indexes/python-course-chroma
uv --directory guardrails-llm-deployment run guardrails-llm visualize --corpus data/python_course_docs.jsonl --course-id python-intro --retriever vector --index-dir indexes/python-course-chroma --mode guardrailed --question "What is declarative knowledge?" --output reports/python_course_rag_demo.html
uv --directory guardrails-llm-deployment run guardrails-llm evaluate --mode baseline --retriever langchain
uv --directory guardrails-llm-deployment run guardrails-llm evaluate --mode guardrailed --retriever langchain --show-results
```

From this package folder:

```bash
./scripts/run_workshop2_demo.sh
uv run python -m guardrails_llm.cli query --mode guardrailed --retriever langchain --question "What is retrieval augmented generation?"
uv run python -m guardrails_llm.cli validate-corpus --corpus data/course_docs.jsonl
uv run python -m guardrails_llm.cli build-index --corpus data/course_docs.jsonl --index-dir indexes/chroma
uv run python -m guardrails_llm.cli query --mode guardrailed --retriever vector --index-dir indexes/chroma --question "What is retrieval augmented generation?"
uv run python -m guardrails_llm.cli normalize-course-corpus --source ../course_corpus/datainmd --output data/python_course_docs.jsonl --course-id python-intro
uv run python -m guardrails_llm.cli build-index --corpus data/python_course_docs.jsonl --index-dir indexes/python-course-chroma
uv run python -m guardrails_llm.cli visualize --corpus data/python_course_docs.jsonl --course-id python-intro --retriever vector --index-dir indexes/python-course-chroma --mode guardrailed --question "What is declarative knowledge?" --output reports/python_course_rag_demo.html
uv run python -m guardrails_llm.cli evaluate --mode baseline --retriever langchain
uv run python -m guardrails_llm.cli evaluate --mode guardrailed --retriever langchain --show-results
```

You can also use the installed console script:

```bash
uv run guardrails-llm evaluate --mode guardrailed --retriever langchain
```

Run the local tests:

```bash
uv run pytest
```

The `--retriever lexical` backend is a dependency-light fallback. The
`--retriever langchain` backend uses LangChain document objects and recursive
text splitting while keeping deterministic local scoring for reproducible
evaluation. The `--retriever vector` backend uses local deterministic hashing
embeddings with a persisted Chroma index for the Workshop 2 baseline demo.

The expected collaborator corpus handoff is normalized JSONL. See
`docs/corpus_contract.md` and validate any delivered corpus before indexing.
The markdown course corpus can be regenerated with `normalize-course-corpus`;
raw PDF/source drops are kept out of git.

## Workshop 2 Status

Current status: the repository has a deterministic toy-corpus prototype, a
normalized Python course corpus, a local Chroma vector index path, and static
HTML visualization for the RAG pipeline.

## Next Implementation Steps

1. Replace the demo corpus with real or self-created course material.
2. Rebuild the Chroma index and refresh vector evaluation reports.
3. Add an optional real LLM adapter.
4. Expand the adversarial JSONL set for prompt injection, privacy, hallucination, and academic-integrity misuse.
5. Use the evaluator output in the Workshop 2 failure analysis.
