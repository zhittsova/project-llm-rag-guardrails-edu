from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import asdict

from .corpus import default_data_path, validate_corpus
from .course_corpus import default_course_output_path, default_course_source_path, normalize_course_corpus
from .evaluation import load_eval_cases, results_to_json, run_evaluation, summarize, write_results_csv
from .guardrail_policy import default_policy_path, load_guardrail_policy
from .judging import judge_results, judgments_to_json, summarize_judgments
from .model_config import MissingModelCredentialError, RemoteModelsNotAllowedError, openai_config_summary
from .pipeline import build_assistant
from .vector import VectorIndexError, build_vector_index, default_index_path
from .visualization import write_rag_visualization


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardrailed RAG learning-assistant prototype")
    parser.add_argument("--corpus", type=Path, default=default_data_path())

    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query", help="Ask one question")
    query_parser.add_argument("--corpus", dest="command_corpus", type=Path)
    query_parser.add_argument("--index-dir", type=Path, default=default_index_path())
    query_parser.add_argument("--course-id", default="guardrails-101")
    query_parser.add_argument("--mode", choices=["baseline", "guardrailed"], default="guardrailed")
    query_parser.add_argument("--retriever", choices=["lexical", "langchain", "vector"], default="lexical")
    _add_embedding_args(query_parser)
    _add_generation_args(query_parser)
    _add_guard_classifier_args(query_parser)
    query_parser.add_argument("--policy", type=Path)
    query_parser.add_argument("--question", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Run JSONL evaluation")
    eval_parser.add_argument("--corpus", dest="command_corpus", type=Path)
    eval_parser.add_argument("--index-dir", type=Path, default=default_index_path())
    eval_parser.add_argument("--course-id", default="guardrails-101")
    eval_parser.add_argument("--mode", choices=["baseline", "guardrailed"], default="guardrailed")
    eval_parser.add_argument("--retriever", choices=["lexical", "langchain", "vector"], default="lexical")
    _add_embedding_args(eval_parser)
    _add_generation_args(eval_parser)
    _add_guard_classifier_args(eval_parser)
    eval_parser.add_argument("--cases", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "eval_cases.jsonl")
    eval_parser.add_argument("--policy", type=Path)
    eval_parser.add_argument("--judge", choices=["none", "heuristic", "openai"], default="none")
    eval_parser.add_argument("--judge-model")
    eval_parser.add_argument("--limit-cases", type=int)
    eval_parser.add_argument("--output-csv", type=Path)
    eval_parser.add_argument("--output-judgments", type=Path)
    eval_parser.add_argument("--show-results", action="store_true")
    eval_parser.add_argument("--show-judgments", action="store_true")

    compare_parser = subparsers.add_parser("compare-guardrails", help="Compare guardrail techniques on one evaluation set")
    compare_parser.add_argument("--corpus", dest="command_corpus", type=Path)
    compare_parser.add_argument("--index-dir", type=Path, default=default_index_path())
    compare_parser.add_argument("--course-id", default="guardrails-101")
    compare_parser.add_argument("--retriever", choices=["lexical", "langchain", "vector"], default="lexical")
    _add_embedding_args(compare_parser)
    _add_generation_args(compare_parser)
    _add_guard_classifier_args(compare_parser)
    compare_parser.add_argument("--cases", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "eval_cases.jsonl")
    compare_parser.add_argument("--policy", type=Path, default=default_policy_path())
    compare_parser.add_argument("--judge", choices=["none", "heuristic", "openai"], default="none")
    compare_parser.add_argument("--judge-model")
    compare_parser.add_argument("--limit-cases", type=int)
    compare_parser.add_argument("--output-json", type=Path)

    index_parser = subparsers.add_parser("build-index", help="Build a local Chroma vector index")
    index_parser.add_argument("--corpus", dest="command_corpus", type=Path)
    index_parser.add_argument("--index-dir", type=Path, default=default_index_path())
    index_parser.add_argument("--chunk-size", type=int, default=650)
    index_parser.add_argument("--chunk-overlap", type=int, default=80)
    _add_embedding_args(index_parser)

    validate_parser = subparsers.add_parser("validate-corpus", help="Validate a corpus JSONL file")
    validate_parser.add_argument("--corpus", dest="command_corpus", type=Path)

    policy_parser = subparsers.add_parser("validate-policy", help="Validate a guardrail policy TOML file")
    policy_parser.add_argument("--policy", type=Path, default=default_policy_path())

    model_config_parser = subparsers.add_parser("model-config", help="Show safe remote-model configuration")
    model_config_parser.add_argument("--provider", choices=["openai"], default="openai")
    model_config_parser.add_argument("--env-file", type=Path)

    course_parser = subparsers.add_parser("normalize-course-corpus", help="Normalize markdown course corpus to JSONL")
    course_parser.add_argument("--source", type=Path, default=default_course_source_path())
    course_parser.add_argument("--output", type=Path, default=default_course_output_path())
    course_parser.add_argument("--course-id", default="python-intro")

    visualize_parser = subparsers.add_parser("visualize", help="Write a static HTML RAG pipeline visualization")
    visualize_parser.add_argument("--corpus", dest="command_corpus", type=Path)
    visualize_parser.add_argument("--index-dir", type=Path, default=default_index_path())
    visualize_parser.add_argument("--course-id", default="guardrails-101")
    visualize_parser.add_argument("--mode", choices=["baseline", "guardrailed"], default="guardrailed")
    visualize_parser.add_argument("--retriever", choices=["lexical", "langchain", "vector"], default="lexical")
    _add_embedding_args(visualize_parser)
    _add_generation_args(visualize_parser)
    _add_guard_classifier_args(visualize_parser)
    visualize_parser.add_argument("--policy", type=Path)
    visualize_parser.add_argument("--question", required=True)
    visualize_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    corpus_path = getattr(args, "command_corpus", None) or args.corpus

    if args.command == "validate-corpus":
        documents = validate_corpus(corpus_path)
        print(json.dumps({"corpus": str(corpus_path), "documents": len(documents)}, indent=2))
        return

    if args.command == "validate-policy":
        policy = load_guardrail_policy(args.policy)
        print(
            json.dumps(
                {
                    "policy": str(args.policy),
                    "input_rules": len(policy.input_rules),
                    "input_similarity_rules": len(policy.input_similarity_rules),
                    "input_fuzzy_rules": len(policy.input_fuzzy_rules),
                    "output_rules": len(policy.output_rules),
                    "output_fuzzy_rules": len(policy.output_fuzzy_rules),
                    "context_rules": len(policy.context_rules),
                    "context_fuzzy_rules": len(policy.context_fuzzy_rules),
                    "blocking_triggers": sorted(policy.blocking_triggers),
                    "allowed_visibility": sorted(policy.allowed_visibility),
                    "require_citations": policy.require_citations,
                },
                indent=2,
            )
        )
        return

    if args.command == "model-config":
        print(json.dumps(openai_config_summary(args.env_file), indent=2))
        return

    if args.command == "normalize-course-corpus":
        stats = normalize_course_corpus(args.source, args.output, course_id=args.course_id)
        print(
            json.dumps(
                {
                    "source_dir": str(stats.source_dir),
                    "output_path": str(stats.output_path),
                    "documents": stats.documents,
                },
                indent=2,
            )
        )
        return

    if args.command == "build-index":
        try:
            stats = build_vector_index(
                corpus_path,
                args.index_dir,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                embedding_provider=args.embedding_provider,
                embedding_model=args.embedding_model,
                allow_remote_models=args.allow_remote_models,
                env_file=args.env_file,
            )
        except (VectorIndexError, RemoteModelsNotAllowedError, MissingModelCredentialError) as exc:
            parser.error(str(exc))
        print(json.dumps(stats.__dict__ | {"corpus": str(stats.corpus), "index_dir": str(stats.index_dir)}, indent=2))
        return

    if args.command == "visualize":
        guardrail_policy = load_guardrail_policy(args.policy) if args.policy else None
        try:
            stats = write_rag_visualization(
                corpus_path=corpus_path,
                output_path=args.output,
                question=args.question,
                mode=args.mode,
                retriever_backend=args.retriever,
                index_dir=args.index_dir,
                course_id=args.course_id,
                guardrail_policy=guardrail_policy,
                embedding_provider=args.embedding_provider,
                embedding_model=args.embedding_model,
                allow_remote_models=args.allow_remote_models,
                env_file=args.env_file,
                generator=args.generator,
                answer_model=args.answer_model,
                guard_classifier=args.guard_classifier,
                classifier_model=args.classifier_model,
            )
        except (VectorIndexError, RemoteModelsNotAllowedError, MissingModelCredentialError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "output_path": str(stats.output_path),
                    "retrieved_chunks": stats.retrieved_chunks,
                },
                indent=2,
            )
        )
        return

    if args.command == "compare-guardrails":
        cases = load_eval_cases(args.cases)
        cases = _limit_cases(cases, args.limit_cases)
        try:
            comparisons = {}
            judge = _build_judge(args)
            for label, mode, policy, classifier, profile in _comparison_scenarios(args):
                comparison_assistant = build_assistant(
                    corpus_path,
                    mode=mode,
                    retriever_backend=args.retriever,
                    index_dir=args.index_dir,
                    course_id=args.course_id,
                    guardrail_policy=policy,
                    embedding_provider=args.embedding_provider,
                    embedding_model=args.embedding_model,
                    allow_remote_models=args.allow_remote_models,
                    env_file=args.env_file,
                    generator=args.generator,
                    answer_model=args.answer_model,
                    guard_classifier=classifier,
                    classifier_model=args.classifier_model,
                )
                comparison_results = run_evaluation(comparison_assistant, cases)
                comparison_summary["results"] = [asdict(result) for result in comparison_results]
                if judge:
                    comparison_summary["judge"] = summarize_judgments(judge_results(cases, comparison_results, judge))
                comparisons[label] = comparison_summary
        except (VectorIndexError, RemoteModelsNotAllowedError, MissingModelCredentialError) as exc:
            parser.error(str(exc))
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
        print(json.dumps(comparisons, indent=2))
        return

    try:
        guardrail_policy = load_guardrail_policy(args.policy) if getattr(args, "policy", None) else None
        assistant = build_assistant(
            corpus_path,
            mode=args.mode,
            retriever_backend=args.retriever,
            index_dir=args.index_dir,
            course_id=args.course_id,
            guardrail_policy=guardrail_policy,
            embedding_provider=getattr(args, "embedding_provider", "hashing"),
            embedding_model=getattr(args, "embedding_model", None),
            allow_remote_models=getattr(args, "allow_remote_models", False),
            env_file=getattr(args, "env_file", None),
            generator=getattr(args, "generator", "extractive"),
            answer_model=getattr(args, "answer_model", None),
            guard_classifier=getattr(args, "guard_classifier", "none"),
            classifier_model=getattr(args, "classifier_model", None),
        )
    except (VectorIndexError, RemoteModelsNotAllowedError, MissingModelCredentialError) as exc:
        parser.error(str(exc))
    if args.command == "query":
        response = assistant.answer(args.question)
        print(json.dumps(response.__dict__, indent=2))
        return

    cases = load_eval_cases(args.cases)
    cases = _limit_cases(cases, getattr(args, "limit_cases", None))
    results = run_evaluation(assistant, cases)
    summary = summarize(results)
    try:
        judge = _build_judge(args)
    except (RemoteModelsNotAllowedError, MissingModelCredentialError) as exc:
        parser.error(str(exc))
    judgments = []
    if judge:
        judgments = judge_results(cases, results, judge)
        summary["judge"] = summarize_judgments(judgments)
    print(json.dumps(summary, indent=2))
    if args.output_csv:
        write_results_csv(results, args.output_csv)
    if args.show_results:
        print(results_to_json(results))
    if args.output_judgments and judgments:
        args.output_judgments.parent.mkdir(parents=True, exist_ok=True)
        args.output_judgments.write_text(judgments_to_json(judgments), encoding="utf-8")
    if args.show_judgments and judgments:
        print(judgments_to_json(judgments))


def _add_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-provider", choices=["hashing", "openai"], default="hashing")
    parser.add_argument("--embedding-model")
    parser.add_argument("--allow-remote-models", action="store_true")
    parser.add_argument("--env-file", type=Path)


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generator", choices=["extractive", "openai"], default="extractive")
    parser.add_argument("--answer-model")


def _add_guard_classifier_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--guard-classifier", choices=["none", "openai"], default="none")
    parser.add_argument("--classifier-model")


def _limit_cases(cases, limit: int | None):
    if limit is None:
        return cases
    return cases[: max(limit, 0)]


def _build_judge(args):
    if getattr(args, "judge", "none") == "none":
        return None
    if args.judge == "heuristic":
        from .judging import HeuristicJudge

        return HeuristicJudge()
    if args.judge == "openai":
        from .model_config import DEFAULT_OPENAI_JUDGE_MODEL, OpenAIModelConfig
        from .openai_models import OpenAIJudge

        return OpenAIJudge(
            OpenAIModelConfig(
                judge_model=args.judge_model or DEFAULT_OPENAI_JUDGE_MODEL,
                allow_remote_models=args.allow_remote_models,
                env_file=args.env_file,
            )
        )
    raise ValueError("judge must be 'none', 'heuristic', or 'openai'")


def _comparison_scenarios(args):
    policy = load_guardrail_policy(args.policy)
    scenarios = [
        (
            "baseline",
            "baseline",
            None,
            "none",
            {
                "technique": "RAG without guardrails",
                "guardrail_layers": [],
                "latency_expected": "lowest",
                "robustness_expected": "lowest",
                "implementation_effort": "low",
            },
        ),
        (
            "default_guardrails",
            "guardrailed",
            None,
            "none",
            {
                "technique": "normalized rules + fuzzy checks + metadata retrieval filters",
                "guardrail_layers": [
                    "text_normalization",
                    "regex_input",
                    "fuzzy_input",
                    "metadata_filter",
                    "context_sanitization",
                    "output_check",
                ],
                "latency_expected": "low-medium",
                "robustness_expected": "medium-high_on_typos",
                "implementation_effort": "medium",
            },
        ),
        (
            "hybrid_policy_guardrails",
            "guardrailed",
            policy,
            "none",
            {
                "technique": "configurable policy + normalized rules + fuzzy checks + embedding similarity examples",
                "guardrail_layers": [
                    "policy_file",
                    "text_normalization",
                    "regex_input",
                    "fuzzy_input",
                    "embedding_similarity_input",
                    "metadata_filter",
                    "context_sanitization",
                    "output_check",
                ],
                "latency_expected": "low-medium",
                "robustness_expected": "medium-high",
                "implementation_effort": "medium-high",
            },
        ),
    ]
    if args.guard_classifier != "none":
        scenarios.append(
            (
                "model_classifier_guardrails",
                "guardrailed",
                policy,
                args.guard_classifier,
                {
                    "technique": "hybrid policy + model classifier for ambiguous prompts",
                    "guardrail_layers": [
                        "policy_file",
                        "text_normalization",
                        "regex_input",
                        "fuzzy_input",
                        "embedding_similarity_input",
                        "model_classifier",
                        "metadata_filter",
                        "context_sanitization",
                        "output_check",
                    ],
                    "latency_expected": "highest",
                    "robustness_expected": "highest_on_paraphrases",
                    "implementation_effort": "high",
                },
            )
        )
    return scenarios


if __name__ == "__main__":
    main()
