# Workshop 2 Demo

This demo can use either the synthetic six-document corpus in
`data/course_docs.jsonl` or the normalized Python course corpus in
`data/python_course_docs.jsonl`.

Scope and readiness are written up in `docs/workshop2_requirements.md`.
The 4-slide presentation deck is `docs/workshop2_slides.html`.
The RAG-focused 10-slide deck is `docs/workshop2_rag_slides.html`.

## One-Command Demo

From the `AMT/` repository root:

```bash
./scripts/run_workshop2_demo.sh
```

From this package folder:

```bash
./scripts/run_workshop2_demo.sh
```

The script validates `data/python_course_docs.jsonl`, rebuilds
`indexes/python-course-chroma`, runs one guardrailed vector query, and writes
`reports/python_course_rag_demo.html`. It then switches to the synthetic
failure-analysis corpus and shows what slips through baseline RAG without the
guardrails.

To try another question:

```bash
QUESTION="What is procedural knowledge?" ./scripts/run_workshop2_demo.sh
```

## Baseline RAG Setup

```bash
uv run guardrails-llm validate-corpus --corpus data/course_docs.jsonl
uv run guardrails-llm build-index --corpus data/course_docs.jsonl --index-dir indexes/chroma
```

For the Python course corpus:

```bash
uv run guardrails-llm normalize-course-corpus --source ../course_corpus/datainmd --output data/python_course_docs.jsonl --course-id python-intro
uv run guardrails-llm validate-corpus --corpus data/python_course_docs.jsonl
uv run guardrails-llm build-index --corpus data/python_course_docs.jsonl --index-dir indexes/python-course-chroma
```

## Course-Material Q&A

```bash
uv run guardrails-llm query --mode guardrailed --retriever vector --index-dir indexes/chroma --question "What is retrieval augmented generation?"
```

Expected behavior: the assistant answers from `rag-basics` and cites the course
source.

For the Python course corpus:

```bash
uv run guardrails-llm query --mode guardrailed --course-id python-intro --retriever vector --corpus data/python_course_docs.jsonl --index-dir indexes/python-course-chroma --question "What is declarative knowledge?"
```

Expected behavior: the assistant answers from Lecture 1 and cites the lecture.

## HTML Visualization

```bash
uv run guardrails-llm visualize --corpus data/python_course_docs.jsonl --course-id python-intro --retriever vector --index-dir indexes/python-course-chroma --mode guardrailed --question "What is declarative knowledge?" --output reports/python_course_rag_demo.html
```

Open `reports/python_course_rag_demo.html` in a browser. The page shows the question,
pipeline stages, retrieved chunks, final answer, citations, guard triggers, and
latency.

## Baseline Failure Contrast

The live Python course corpus is public course material. It is good for the normal
RAG demo, but it does not contain private documents. The controlled safety failures
are demonstrated with `data/course_docs.jsonl`.

The one-command demo shows two concrete baseline failures:

- baseline retrieves `private-roster` for a sensitive-data question;
- baseline repeats an injected instruction from retrieved assignment text.

The guardrailed run blocks the sensitive-data request before retrieval and
sanitizes the injected retrieved text before answering.

## Unsupported Question

```bash
uv run guardrails-llm query --mode guardrailed --retriever vector --index-dir indexes/chroma --question "Which GPU model was used to train the assistant?"
```

Expected behavior: the assistant abstains because the corpus has no supporting
course evidence.

## Failure Analysis Run

```bash
uv run guardrails-llm evaluate --mode baseline --retriever vector --index-dir indexes/chroma --output-csv reports/baseline_vector_results.csv
uv run guardrails-llm evaluate --mode guardrailed --retriever vector --index-dir indexes/chroma --output-csv reports/guardrailed_vector_results.csv
```

Current synthetic-corpus results:

- Baseline vector: `3/12`.
- Guardrailed vector: `12/12`.

The Python course corpus should be used for the live corpus demo. The synthetic corpus
remains useful as a stable regression check.
