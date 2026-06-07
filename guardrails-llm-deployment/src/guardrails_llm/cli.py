from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus import default_data_path, validate_corpus
from .evaluation import load_eval_cases, results_to_json, run_evaluation, summarize, write_results_csv
from .pipeline import build_assistant
from .vector import build_vector_index, default_index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardrailed RAG learning-assistant prototype")
    parser.add_argument("--corpus", type=Path, default=default_data_path())

    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query", help="Ask one question")
    query_parser.add_argument("--corpus", dest="command_corpus", type=Path)
    query_parser.add_argument("--index-dir", type=Path, default=default_index_path())
    query_parser.add_argument("--mode", choices=["baseline", "guardrailed"], default="guardrailed")
    query_parser.add_argument("--retriever", choices=["lexical", "langchain", "vector"], default="lexical")
    query_parser.add_argument("--question", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Run JSONL evaluation")
    eval_parser.add_argument("--corpus", dest="command_corpus", type=Path)
    eval_parser.add_argument("--index-dir", type=Path, default=default_index_path())
    eval_parser.add_argument("--mode", choices=["baseline", "guardrailed"], default="guardrailed")
    eval_parser.add_argument("--retriever", choices=["lexical", "langchain", "vector"], default="lexical")
    eval_parser.add_argument("--cases", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "eval_cases.jsonl")
    eval_parser.add_argument("--output-csv", type=Path)
    eval_parser.add_argument("--show-results", action="store_true")

    index_parser = subparsers.add_parser("build-index", help="Build a local Chroma vector index")
    index_parser.add_argument("--corpus", dest="command_corpus", type=Path)
    index_parser.add_argument("--index-dir", type=Path, default=default_index_path())
    index_parser.add_argument("--chunk-size", type=int, default=650)
    index_parser.add_argument("--chunk-overlap", type=int, default=80)

    validate_parser = subparsers.add_parser("validate-corpus", help="Validate a corpus JSONL file")
    validate_parser.add_argument("--corpus", dest="command_corpus", type=Path)

    args = parser.parse_args()
    corpus_path = args.command_corpus or args.corpus

    if args.command == "validate-corpus":
        documents = validate_corpus(corpus_path)
        print(json.dumps({"corpus": str(corpus_path), "documents": len(documents)}, indent=2))
        return

    if args.command == "build-index":
        stats = build_vector_index(
            corpus_path,
            args.index_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(json.dumps(stats.__dict__ | {"corpus": str(stats.corpus), "index_dir": str(stats.index_dir)}, indent=2))
        return

    assistant = build_assistant(
        corpus_path,
        mode=args.mode,
        retriever_backend=args.retriever,
        index_dir=args.index_dir,
    )
    if args.command == "query":
        response = assistant.answer(args.question)
        print(json.dumps(response.__dict__, indent=2))
        return

    cases = load_eval_cases(args.cases)
    results = run_evaluation(assistant, cases)
    print(json.dumps(summarize(results), indent=2))
    if args.output_csv:
        write_results_csv(results, args.output_csv)
    if args.show_results:
        print(results_to_json(results))


if __name__ == "__main__":
    main()
