from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from .corpus import default_data_path, validate_corpus
from .course_corpus import default_course_output_path, default_course_source_path, normalize_course_corpus
from .embeddings import CachedEmbedder, create_embedder
from .evaluation import (
    load_eval_cases,
    results_to_json,
    run_evaluation,
    select_eval_split,
    summarize,
    write_results_csv,
)
from .guardrail_policy import GuardrailPolicy, default_policy_path, load_guardrail_policy
from .guard_text import normalize_guard_text
from .judging import judge_results, judgments_to_json, summarize_judgments
from .model_calibration import (
    DEFAULT_CALIBRATION_SOURCE_CASES,
    DEFAULT_CALIBRATION_SOURCE_RESULTS,
    DEFAULT_CLASSIFIER_CALIBRATION_CASES,
    DEFAULT_CLASSIFIER_CALIBRATION_PREDICTIONS,
    DEFAULT_JUDGE_CALIBRATION_CASES,
    DEFAULT_JUDGE_CALIBRATION_PREDICTIONS,
    run_local_model_calibration,
)
from .model_config import (
    MissingModelCredentialError,
    RemoteModelCallError,
    RemoteModelsNotAllowedError,
    openai_config_summary,
)
from .pipeline import build_assistant
from .retrieval_benchmark import run_local_retrieval_benchmark
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
    _add_guard_embedding_args(query_parser)
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
    _add_guard_embedding_args(eval_parser)
    eval_parser.add_argument("--judge", choices=["none", "heuristic", "openai"], default="none")
    eval_parser.add_argument("--judge-model")
    eval_parser.add_argument("--limit-cases", type=int)
    eval_parser.add_argument("--case-split", choices=["all", "calibration", "validation"], default="all")
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
    _add_guard_embedding_args(compare_parser)
    compare_parser.add_argument("--judge", choices=["none", "heuristic", "openai"], default="none")
    compare_parser.add_argument("--judge-model")
    compare_parser.add_argument("--limit-cases", type=int)
    compare_parser.add_argument("--case-split", choices=["all", "calibration", "validation"], default="all")
    compare_parser.add_argument("--output-json", type=Path)
    compare_parser.add_argument("--output-results-json", type=Path)

    retrieval_benchmark_parser = subparsers.add_parser(
        "benchmark-retrieval",
        help="Compare local lexical and hashing-vector retrieval quality",
    )
    retrieval_benchmark_parser.add_argument(
        "--corpus",
        dest="command_corpus",
        type=Path,
    )
    retrieval_benchmark_parser.add_argument(
        "--cases",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "data"
            / "retrieval_cases_milestone3_v1.jsonl"
        ),
    )
    retrieval_benchmark_parser.add_argument(
        "--index-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "indexes"
            / "retrieval-benchmark-v1"
        ),
    )
    retrieval_benchmark_parser.add_argument("--chunk-size", type=int, default=650)
    retrieval_benchmark_parser.add_argument("--chunk-overlap", type=int, default=80)
    retrieval_benchmark_parser.add_argument("--top-k", type=int, default=3)
    retrieval_benchmark_parser.add_argument("--output-json", type=Path)
    retrieval_benchmark_parser.add_argument("--output-results-json", type=Path)

    calibration_parser = subparsers.add_parser(
        "evaluate-model-calibration",
        help="Replay local classifier and judge predictions against human labels",
    )
    calibration_parser.add_argument(
        "--classifier-cases",
        type=Path,
        default=DEFAULT_CLASSIFIER_CALIBRATION_CASES,
    )
    calibration_parser.add_argument(
        "--classifier-predictions",
        type=Path,
        default=DEFAULT_CLASSIFIER_CALIBRATION_PREDICTIONS,
    )
    calibration_parser.add_argument(
        "--judge-cases",
        type=Path,
        default=DEFAULT_JUDGE_CALIBRATION_CASES,
    )
    calibration_parser.add_argument(
        "--judge-predictions",
        type=Path,
        default=DEFAULT_JUDGE_CALIBRATION_PREDICTIONS,
    )
    calibration_parser.add_argument(
        "--source-cases",
        type=Path,
        default=DEFAULT_CALIBRATION_SOURCE_CASES,
    )
    calibration_parser.add_argument(
        "--source-results",
        type=Path,
        default=DEFAULT_CALIBRATION_SOURCE_RESULTS,
    )
    calibration_parser.add_argument("--output-json", type=Path)

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
    _add_guard_embedding_args(visualize_parser)
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

    if args.command == "benchmark-retrieval":
        try:
            summaries, details = run_local_retrieval_benchmark(
                corpus_path=corpus_path,
                cases_path=args.cases,
                index_dir=args.index_dir,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                top_k=args.top_k,
            )
        except (ValueError, VectorIndexError) as exc:
            parser.error(str(exc))
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(summaries, indent=2),
                encoding="utf-8",
            )
        if args.output_results_json:
            args.output_results_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_results_json.write_text(
                json.dumps(details, indent=2),
                encoding="utf-8",
            )
        print(json.dumps(summaries, indent=2))
        return

    if args.command == "evaluate-model-calibration":
        try:
            calibration = run_local_model_calibration(
                classifier_cases_path=args.classifier_cases,
                classifier_predictions_path=args.classifier_predictions,
                judge_cases_path=args.judge_cases,
                judge_predictions_path=args.judge_predictions,
                source_cases_path=args.source_cases,
                source_results_path=args.source_results,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(calibration, indent=2),
                encoding="utf-8",
            )
        print(json.dumps(calibration, indent=2))
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
        except (
            VectorIndexError,
            RemoteModelsNotAllowedError,
            MissingModelCredentialError,
            RemoteModelCallError,
        ) as exc:
            parser.error(str(exc))
        print(json.dumps(stats.__dict__ | {"corpus": str(stats.corpus), "index_dir": str(stats.index_dir)}, indent=2))
        return

    if args.command == "visualize":
        try:
            guardrail_policy = _load_guardrail_policy(args)
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
        except (
            VectorIndexError,
            RemoteModelsNotAllowedError,
            MissingModelCredentialError,
            RemoteModelCallError,
        ) as exc:
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
        cases = select_eval_split(cases, args.case_split)
        cases = _limit_cases(cases, args.limit_cases)
        try:
            comparisons = {}
            comparison_details = {}
            judge = _build_judge(args)
            retrieval_embedder, retrieval_preload = _preload_retrieval_embedder(args, cases)
            guardrail_policy, guard_preload = _load_comparison_policy(args, cases)
            for label, mode, policy, classifier, profile in _comparison_scenarios(
                args,
                guardrail_policy,
            ):
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
                    retrieval_embedder=retrieval_embedder,
                )
                comparison_results = run_evaluation(comparison_assistant, cases)
                comparison_details[label] = [
                    asdict(result) for result in comparison_results
                ]
                comparison_summary = profile | summarize(comparison_results)
                comparison_summary["eval_split"] = args.case_split
                preloads = {}
                if retrieval_preload:
                    preloads["retrieval"] = retrieval_preload
                if guard_preload and label in {"hybrid_policy_guardrails", "model_classifier_guardrails"}:
                    preloads["guard_similarity"] = guard_preload
                if preloads:
                    preload_ms = sum(float(stats["latency_ms"]) for stats in preloads.values())
                    comparison_summary["embedding_preload"] = preloads
                    comparison_summary["avg_batch_amortized_latency_ms"] = round(
                        float(comparison_summary["avg_latency_ms"]) + preload_ms / max(len(cases), 1),
                        2,
                    )
                    comparison_summary["latency_scope"] = "pipeline_after_batch_preload"
                if judge:
                    comparison_summary["judge"] = summarize_judgments(judge_results(cases, comparison_results, judge))
                comparisons[label] = comparison_summary
        except (
            VectorIndexError,
            RemoteModelsNotAllowedError,
            MissingModelCredentialError,
            RemoteModelCallError,
        ) as exc:
            parser.error(str(exc))
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
        if args.output_results_json:
            args.output_results_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_results_json.write_text(
                json.dumps(comparison_details, indent=2),
                encoding="utf-8",
            )
        print(json.dumps(comparisons, indent=2))
        return

    try:
        guardrail_policy = _load_guardrail_policy(args)
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
    except (
        VectorIndexError,
        RemoteModelsNotAllowedError,
        MissingModelCredentialError,
        RemoteModelCallError,
    ) as exc:
        parser.error(str(exc))
    if args.command == "query":
        try:
            response = assistant.answer(args.question)
        except RemoteModelCallError as exc:
            parser.error(str(exc))
        print(json.dumps(response.__dict__, indent=2))
        return

    cases = load_eval_cases(args.cases)
    cases = select_eval_split(cases, args.case_split)
    cases = _limit_cases(cases, getattr(args, "limit_cases", None))
    try:
        results = run_evaluation(assistant, cases)
    except RemoteModelCallError as exc:
        parser.error(str(exc))
    summary = summarize(results)
    summary["eval_split"] = args.case_split
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


def _add_guard_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--guard-embedding-provider", choices=["hashing", "openai"], default="hashing")
    parser.add_argument("--guard-embedding-model")


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


def _load_guardrail_policy(args):
    policy_path = getattr(args, "policy", None)
    if policy_path is None:
        return None
    similarity_embedder = create_embedder(
        args.guard_embedding_provider,
        model=args.guard_embedding_model,
        allow_remote_models=args.allow_remote_models,
        env_file=args.env_file,
    )
    return load_guardrail_policy(policy_path, similarity_embedder=similarity_embedder)


def _preload_retrieval_embedder(args, cases):
    if args.retriever != "vector":
        return None, None
    embedder = CachedEmbedder(
        create_embedder(
            args.embedding_provider,
            model=args.embedding_model,
            allow_remote_models=args.allow_remote_models,
            env_file=args.env_file,
        )
    )
    stats = _preload_embeddings(
        embedder,
        [case.question for case in cases],
        provider=args.embedding_provider,
    )
    return embedder, stats


def _load_comparison_policy(args, cases):
    policy_path = getattr(args, "policy", None)
    if policy_path is None:
        return None, None
    embedder = CachedEmbedder(
        create_embedder(
            args.guard_embedding_provider,
            model=args.guard_embedding_model,
            allow_remote_models=args.allow_remote_models,
            env_file=args.env_file,
        )
    )
    policy = load_guardrail_policy(policy_path, similarity_embedder=embedder)
    texts = [
        normalize_guard_text(example)
        for rule in policy.input_similarity_rules
        for example in rule.examples
    ]
    texts.extend(normalize_guard_text(case.question) for case in cases)
    stats = _preload_embeddings(
        embedder,
        texts,
        provider=args.guard_embedding_provider,
    )
    return policy, stats


def _preload_embeddings(embedder: CachedEmbedder, texts: list[str], *, provider: str) -> dict[str, object]:
    calls_before = embedder.api_call_count
    started_at = perf_counter()
    embedder.embed_many(texts)
    latency_ms = (perf_counter() - started_at) * 1000
    calls_after = embedder.api_call_count
    provider_calls = None
    if calls_before is not None and calls_after is not None:
        provider_calls = calls_after - calls_before
    return {
        "provider": provider,
        "model": embedder.model_name,
        "texts": len(texts),
        "unique_texts": embedder.cached_texts,
        "provider_calls": provider_calls,
        "latency_ms": round(latency_ms, 2),
    }


def _comparison_scenarios(args, policy):
    regex_policy = replace(
        GuardrailPolicy.default(),
        input_similarity_rules=(),
        input_fuzzy_rules=(),
        output_fuzzy_rules=(),
        context_fuzzy_rules=(),
    )
    fuzzy_policy = replace(
        policy,
        input_rules=(),
        input_similarity_rules=(),
        output_rules=(),
        context_rules=(),
    )
    embedding_policy = replace(
    policy,
    input_rules=(),
    input_fuzzy_rules=(),
    output_rules=(),
    output_fuzzy_rules=(),
    context_rules=(),
    context_fuzzy_rules=(),
    )

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
            "normalized_regex_guardrails",
            "guardrailed",
            regex_policy,
            "none",
            {
                "technique": "normalized regex rules + metadata retrieval filters",
                "guardrail_layers": [
                    "text_normalization",
                    "regex_input",
                    "metadata_filter",
                    "context_sanitization",
                    "output_check",
                ],
                "latency_expected": "low",
                "robustness_expected": "medium_on_known_patterns",
                "implementation_effort": "low-medium",
            },
        ),
          (
            "fuzzy_only_guardrails",
            "guardrailed",
            fuzzy_policy,
            "none",
            {
                "technique": "fuzzy rules + metadata retrieval filters",
                "guardrail_layers": [
                    "text_normalization",
                    "fuzzy_input",
                    "metadata_filter",
                    "context_fuzzy_sanitization",
                    "output_fuzzy_check",
                ],
                "latency_expected": "low-medium",
                "robustness_expected": "medium_on_typos_and_near_matches",
                "implementation_effort": "medium",
            },
        ),
         (
            "embedding_only_guardrails",
            "guardrailed",
            embedding_policy,
            "none",
            {
                "technique": "embedding similarity rules + metadata retrieval filters",
                "guardrail_layers": [
                    "embedding_similarity_input",
                    "metadata_filter",
                ],
                "latency_expected": "low-medium",
                "robustness_expected": "medium_on_semantic_variants",
                "implementation_effort": "medium-high",
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
