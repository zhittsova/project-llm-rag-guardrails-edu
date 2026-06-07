# Project Plan

## Research Question

How can layered guardrails reduce safety, privacy, and hallucination risks in an LLM-based learning assistant without making the assistant unusable?

## Prototype Scope

- RAG-based text Q&A assistant for one course corpus.
- LangChain-backed document loading/chunking path plus a dependency-light lexical fallback.
- Input validation for prompt injection, PII, off-topic, and academic-integrity misuse.
- Retrieval constraints using course ID and visibility metadata.
- Retrieved-context sanitization because retrieved documents are treated as untrusted data.
- Output validation for groundedness, citations, moderation, and leakage checks.
- Evaluation with the same JSONL test set for baseline RAG and guardrailed RAG.

## Architecture

```text
User query
  -> input guards
  -> constrained retrieval
  -> prompt/answer generation
  -> output guards
  -> final answer with citations and logs
```

## Evaluation Metrics

- Attack success rate for prompt injection.
- Unsupported answer rate for hallucination and grounding.
- Leakage rate for private or sensitive data.
- Inappropriate compliance rate for academic-integrity misuse.
- Refusal rate on normal learning questions.
- Latency and guard-trigger counts.

## Workshop 2 Demo Target

Run the same 12-case evaluation set against:

- `baseline + langchain retriever`
- `guardrailed + langchain retriever`

The expected story is that the baseline handles normal course questions but fails adversarial, privacy, and misuse cases, while the guardrailed pipeline blocks or safely redirects those cases with a small latency overhead.

## Milestones

- Workshop 1: problem framing, threat model, architecture, evaluation design.
- Workshop 2: baseline RAG, first guardrails, failure analysis, demo.
- Final submission: repository, final report, setup docs, and best-practice checklist.
