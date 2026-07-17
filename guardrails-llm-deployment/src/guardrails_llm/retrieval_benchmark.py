from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .corpus import load_documents
from .langchain_rag import langchain_chunk_documents
from .retrieval import LexicalRetriever
from .retrieval_evaluation import (
    load_retrieval_cases,
    run_retrieval_evaluation,
    summarize_retrieval,
    validate_retrieval_cases,
)
from .vector import VectorRetriever, build_vector_index


def run_local_retrieval_benchmark(
    *,
    corpus_path: Path,
    cases_path: Path,
    index_dir: Path,
    chunk_size: int = 650,
    chunk_overlap: int = 80,
    top_k: int = 3,
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]]]:
    documents = load_documents(corpus_path)
    cases = load_retrieval_cases(cases_path)
    validate_retrieval_cases(cases, documents)

    chunks = langchain_chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    lexical = LexicalRetriever(chunks)
    build_vector_index(
        corpus_path,
        index_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_provider="hashing",
    )
    hashing_vector = VectorRetriever(
        index_dir,
        embedding_provider="hashing",
    )

    scenarios = {
        "lexical": (
            lexical,
            "TF-IDF-style lexical term matching",
        ),
        "hashing_vector": (
            hashing_vector,
            "local deterministic hashing-vector similarity",
        ),
    }
    summaries: dict[str, dict[str, object]] = {}
    details: dict[str, list[dict[str, object]]] = {}
    for label, (retriever, technique) in scenarios.items():
        results = run_retrieval_evaluation(retriever, cases, top_k=top_k)
        summaries[label] = {
            "technique": technique,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k,
            **summarize_retrieval(results),
        }
        details[label] = [asdict(result) for result in results]
    return summaries, details
