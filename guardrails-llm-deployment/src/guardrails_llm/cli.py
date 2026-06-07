from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus import default_data_path
from .evaluation import load_eval_cases, results_to_json, run_evaluation, summarize, write_results_csv
from .pipeline import build_assistant


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardrailed RAG learning-assistant prototype")
    parser.add_argument("--corpus", type=Path, default=default_data_path())

    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query", help="Ask one question")
    query_parser.add_argument("--mode", choices=["baseline", "guardrailed"], default="guardrailed")
    query_parser.add_argument("--retriever", choices=["lexical", "langchain"], default="lexical")
    query_parser.add_argument("--question", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Run JSONL evaluation")
    eval_parser.add_argument("--mode", choices=["baseline", "guardrailed"], default="guardrailed")
    eval_parser.add_argument("--retriever", choices=["lexical", "langchain"], default="lexical")
    eval_parser.add_argument("--cases", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "eval_cases.jsonl")
    eval_parser.add_argument("--output-csv", type=Path)
    eval_parser.add_argument("--show-results", action="store_true")

    args = parser.parse_args()

    assistant = build_assistant(args.corpus, mode=args.mode, retriever_backend=args.retriever)
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
