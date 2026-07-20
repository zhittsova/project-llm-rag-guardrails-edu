from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from .confidence_intervals import bootstrap_confidence_intervals
from .corpus import default_data_path, validate_corpus
from .bge_evaluation import run_bge_common_split_evaluation
from .course_corpus import default_course_output_path, default_course_source_path, normalize_course_corpus
from .e2e_capture import (
    evaluate_calibration_e2e_capture,
    run_calibration_e2e_capture,
)
from .embeddings import CachedEmbedder, create_embedder
from .evaluation import (
    EvalCase,
    results_to_json,
    run_evaluation,
    select_eval_split,
    summarize,
    write_results_csv,
)
from .evaluation_dataset import DatasetValidationError, load_evaluation_cases_for_run
from .final_evidence import (
    FinalEvidenceError,
    assess_final_readiness_from_files,
    seal_runtime_configuration,
    write_calibration_evidence,
)
from .guardrail_policy import GuardrailPolicy, default_policy_path, load_guardrail_policy
from .guard_text import normalize_guard_text
from .holdout_review import (
    finalize_holdout_review,
    holdout_review_status,
    prepare_holdout_review,
    reconcile_holdout_review,
)
from .inhouse_experiment import (
    evaluate_v2_classifier_capture,
    prepare_inhouse_bge,
    run_v2_classifier_capture,
)
from .judging import judge_results, judgments_to_json, summarize_judgments
from .judge_study import (
    JUDGE_SPLITS,
    evaluate_judge_study_models,
    finalize_human_ground_truth,
    prepare_judge_study,
    reconcile_human_annotations,
    validate_annotation_file,
)
from .judge_study_capture import run_judge_study_capture
from .review_server import serve_review_ui
from .reconciliation_server import serve_reconciliation_ui
from .review_recommendations import prepare_review_recommendations
from .model_calibration import (
    DEFAULT_CALIBRATION_SOURCE_CASES,
    DEFAULT_CALIBRATION_SOURCE_RESULTS,
    DEFAULT_CLASSIFIER_CALIBRATION_CASES,
    DEFAULT_CLASSIFIER_CALIBRATION_PREDICTIONS,
    DEFAULT_JUDGE_CALIBRATION_CASES,
    DEFAULT_JUDGE_CALIBRATION_PREDICTIONS,
    run_local_model_calibration,
)
from .model_capture import (
    DEFAULT_CAPTURE_MANIFEST_OUTPUT,
    DEFAULT_CLASSIFIER_CAPTURE_OUTPUT,
    DEFAULT_JUDGE_CAPTURE_OUTPUT,
    run_model_calibration_capture,
)
from .model_config import (
    DEFAULT_OPENAI_CLASSIFIER_MODEL,
    DEFAULT_OPENAI_JUDGE_MODEL,
    MissingModelCredentialError,
    OpenAIModelConfig,
    RemoteModelCallError,
    RemoteModelsNotAllowedError,
    openai_config_summary,
)
from .model_profiles import (
    INHOUSE_EVIDENCE_MIN_SCORE,
    INHOUSE_POLICY_CONTEXT_MIN_SCORE,
    INHOUSE_POLICY_CONTEXT_TOP_K,
    InHouseEndpointError,
    MODEL_PROFILES,
    apply_model_profile,
    model_profile_summary,
)
from .pipeline import build_assistant
from .retrieval_benchmark import run_local_retrieval_benchmark
from .vector import VectorIndexError, build_vector_index, default_index_path
from .visualization import write_rag_visualization
from .workshop3_demo import write_workshop3_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardrailed RAG learning-assistant prototype")
    parser.add_argument("--corpus", type=Path, default=default_data_path())

    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query", help="Ask one question")
    _add_profile_arg(query_parser)
    query_parser.add_argument("--corpus", dest="command_corpus", type=Path)
    query_parser.add_argument("--index-dir", type=Path, default=default_index_path())
    query_parser.add_argument("--course-id", default="guardrails-101")
    query_parser.add_argument("--mode", choices=["baseline", "guardrailed"], default="guardrailed")
    query_parser.add_argument("--retriever", choices=["lexical", "langchain", "vector"], default="lexical")
    _add_embedding_args(query_parser)
    _add_generation_args(query_parser)
    _add_guard_classifier_args(query_parser)
    _add_grounding_args(query_parser)
    query_parser.add_argument("--policy", type=Path)
    _add_guard_embedding_args(query_parser)
    query_parser.add_argument("--question", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Run JSONL evaluation")
    _add_profile_arg(eval_parser)
    eval_parser.add_argument("--corpus", dest="command_corpus", type=Path)
    eval_parser.add_argument("--index-dir", type=Path, default=default_index_path())
    eval_parser.add_argument("--course-id", default="guardrails-101")
    eval_parser.add_argument("--mode", choices=["baseline", "guardrailed"], default="guardrailed")
    eval_parser.add_argument("--retriever", choices=["lexical", "langchain", "vector"], default="lexical")
    _add_embedding_args(eval_parser)
    _add_generation_args(eval_parser)
    _add_guard_classifier_args(eval_parser)
    _add_grounding_args(eval_parser)
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
    _add_profile_arg(compare_parser)
    compare_parser.add_argument("--corpus", dest="command_corpus", type=Path)
    compare_parser.add_argument("--index-dir", type=Path, default=default_index_path())
    compare_parser.add_argument("--course-id", default="guardrails-101")
    compare_parser.add_argument("--retriever", choices=["lexical", "langchain", "vector"], default="lexical")
    _add_embedding_args(compare_parser)
    _add_generation_args(compare_parser)
    _add_guard_classifier_args(compare_parser)
    _add_grounding_args(compare_parser)
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

    capture_parser = subparsers.add_parser(
        "capture-model-calibration",
        help="Capture gated remote classifier and judge predictions",
    )
    _add_profile_arg(capture_parser)
    capture_parser.add_argument(
        "--component",
        choices=["classifier", "judge", "both"],
        default="both",
    )
    capture_parser.add_argument(
        "--classifier-cases",
        type=Path,
        default=DEFAULT_CLASSIFIER_CALIBRATION_CASES,
    )
    capture_parser.add_argument(
        "--judge-cases",
        type=Path,
        default=DEFAULT_JUDGE_CALIBRATION_CASES,
    )
    capture_parser.add_argument(
        "--source-cases",
        type=Path,
        default=DEFAULT_CALIBRATION_SOURCE_CASES,
    )
    capture_parser.add_argument(
        "--source-results",
        type=Path,
        default=DEFAULT_CALIBRATION_SOURCE_RESULTS,
    )
    capture_parser.add_argument(
        "--classifier-output",
        type=Path,
        default=DEFAULT_CLASSIFIER_CAPTURE_OUTPUT,
    )
    capture_parser.add_argument(
        "--judge-output",
        type=Path,
        default=DEFAULT_JUDGE_CAPTURE_OUTPUT,
    )
    capture_parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DEFAULT_CAPTURE_MANIFEST_OUTPUT,
    )
    capture_parser.add_argument("--classifier-model")
    capture_parser.add_argument("--judge-model")
    capture_parser.add_argument("--limit-cases", type=int)
    capture_parser.add_argument(
        "--selection-strategy",
        choices=["stratified", "head"],
        default="stratified",
    )
    capture_parser.add_argument("--allow-remote-models", action="store_true")
    capture_parser.add_argument("--env-file", type=Path)

    v2_classifier_parser = subparsers.add_parser(
        "capture-v2-classifier",
        help="Capture the resumable 600-case in-house classifier benchmark",
    )
    v2_classifier_parser.add_argument(
        "--profile",
        choices=MODEL_PROFILES,
        default="inhouse",
    )
    v2_classifier_parser.add_argument(
        "--development-cases",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "eval_cases_milestone3_v2_development.jsonl",
    )
    v2_classifier_parser.add_argument(
        "--calibration-cases",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "eval_cases_milestone3_v2_calibration.jsonl",
    )
    v2_classifier_parser.add_argument(
        "--course-corpus",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "python_course_docs.jsonl",
    )
    v2_classifier_parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "guardrail_policy_bge_m3.toml",
    )
    v2_classifier_parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports"
        / "inhouse_classifier_v2_predictions.jsonl",
    )
    v2_classifier_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports"
        / "inhouse_classifier_v2_manifest.json",
    )
    v2_classifier_parser.add_argument("--classifier-model")
    v2_classifier_parser.add_argument("--limit-cases", type=int)
    v2_classifier_parser.add_argument("--max-concurrency", type=int, default=1)
    v2_classifier_parser.add_argument("--retry-failures", action="store_true")
    v2_classifier_parser.add_argument("--allow-remote-models", action="store_true")
    v2_classifier_parser.add_argument("--env-file", type=Path)

    v2_classifier_eval_parser = subparsers.add_parser(
        "evaluate-v2-classifier",
        help="Evaluate saved v2 classifier predictions without remote calls",
    )
    v2_classifier_eval_parser.add_argument(
        "--development-cases",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "eval_cases_milestone3_v2_development.jsonl",
    )
    v2_classifier_eval_parser.add_argument(
        "--calibration-cases",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "eval_cases_milestone3_v2_calibration.jsonl",
    )
    v2_classifier_eval_parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports"
        / "inhouse_classifier_v2_predictions.jsonl",
    )
    v2_classifier_eval_parser.add_argument("--limit-cases", type=int)
    v2_classifier_eval_parser.add_argument("--output-json", type=Path)

    bge_prepare_parser = subparsers.add_parser(
        "prepare-inhouse-bge",
        help="Cache v2 BGE embeddings and build the real-course Chroma index",
    )
    bge_prepare_parser.add_argument(
        "--profile",
        choices=["inhouse"],
        default="inhouse",
    )
    bge_prepare_parser.add_argument(
        "--development-cases",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "eval_cases_milestone3_v2_development.jsonl",
    )
    bge_prepare_parser.add_argument(
        "--calibration-cases",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "eval_cases_milestone3_v2_calibration.jsonl",
    )
    bge_prepare_parser.add_argument(
        "--course-corpus",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "python_course_docs.jsonl",
    )
    bge_prepare_parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "guardrail_policy_bge_m3.toml",
    )
    bge_prepare_parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "indexes" / "python-course-bge-m3",
    )
    bge_prepare_parser.add_argument("--chunk-size", type=int, default=650)
    bge_prepare_parser.add_argument("--chunk-overlap", type=int, default=80)
    bge_prepare_parser.add_argument("--embedding-model")
    bge_prepare_parser.add_argument("--embedding-cache", type=Path)
    bge_prepare_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports"
        / "inhouse_bge_preparation_manifest.json",
    )
    bge_prepare_parser.add_argument("--allow-remote-models", action="store_true")
    bge_prepare_parser.add_argument("--env-file", type=Path)

    bge_evaluation_parser = subparsers.add_parser(
        "calibrate-inhouse-bge",
        help="Compare BGE and hashing on common development/calibration cases",
    )
    bge_evaluation_parser.add_argument(
        "--profile",
        choices=["inhouse"],
        default="inhouse",
    )
    bge_evaluation_parser.add_argument(
        "--development-cases",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "eval_cases_milestone3_v2_development.jsonl",
    )
    bge_evaluation_parser.add_argument(
        "--calibration-cases",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "eval_cases_milestone3_v2_calibration.jsonl",
    )
    bge_evaluation_parser.add_argument(
        "--course-corpus",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "python_course_docs.jsonl",
    )
    bge_evaluation_parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "guardrail_policy_bge_m3.toml",
    )
    bge_evaluation_parser.add_argument(
        "--bge-index-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "indexes" / "python-course-bge-m3",
    )
    bge_evaluation_parser.add_argument(
        "--hashing-index-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "indexes" / "python-course-hashing-v2",
    )
    bge_evaluation_parser.add_argument("--embedding-model")
    bge_evaluation_parser.add_argument("--embedding-cache", type=Path)
    bge_evaluation_parser.add_argument("--course-id", default="python-intro")
    bge_evaluation_parser.add_argument("--top-k", type=int, default=3)
    bge_evaluation_parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports"
        / "inhouse_bge_common_split_summary.json",
    )
    bge_evaluation_parser.add_argument(
        "--output-details-json",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports"
        / "inhouse_bge_common_split_details.json",
    )
    bge_evaluation_parser.add_argument("--allow-remote-models", action="store_true")
    bge_evaluation_parser.add_argument("--env-file", type=Path)

    e2e_capture_parser = subparsers.add_parser(
        "capture-inhouse-calibration",
        help="Run resumable Qwen-only and complete-hybrid calibration scenarios",
    )
    e2e_capture_parser.add_argument(
        "--profile",
        choices=["inhouse"],
        default="inhouse",
    )
    e2e_capture_parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "eval_cases_milestone3_v2_calibration.jsonl",
    )
    e2e_capture_parser.add_argument(
        "--course-corpus",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "python_course_docs.jsonl",
    )
    e2e_capture_parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "guardrail_policy_bge_m3.toml",
    )
    e2e_capture_parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "indexes" / "python-course-bge-m3",
    )
    e2e_capture_parser.add_argument("--embedding-model")
    e2e_capture_parser.add_argument("--embedding-cache", type=Path)
    e2e_capture_parser.add_argument("--answer-model")
    e2e_capture_parser.add_argument("--classifier-model")
    e2e_capture_parser.add_argument("--entailment-model")
    e2e_capture_parser.add_argument(
        "--evidence-min-score",
        type=_finite_float,
        default=INHOUSE_EVIDENCE_MIN_SCORE,
    )
    e2e_capture_parser.add_argument(
        "--policy-context-top-k",
        type=int,
        default=INHOUSE_POLICY_CONTEXT_TOP_K,
    )
    e2e_capture_parser.add_argument(
        "--policy-context-min-score",
        type=_finite_float,
        default=INHOUSE_POLICY_CONTEXT_MIN_SCORE,
    )
    e2e_capture_parser.add_argument(
        "--entailment-min-confidence",
        type=_unit_float,
        default=0.80,
    )
    e2e_capture_parser.add_argument("--course-id", default="python-intro")
    e2e_capture_parser.add_argument("--limit-cases", type=int)
    e2e_capture_parser.add_argument("--case-id", dest="case_ids", action="append")
    e2e_capture_parser.add_argument("--max-concurrency", type=int, default=1)
    e2e_capture_parser.add_argument("--retry-failures", action="store_true")
    e2e_capture_parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports"
        / "inhouse_calibration_e2e.jsonl",
    )
    e2e_capture_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports"
        / "inhouse_calibration_e2e_manifest.json",
    )
    e2e_capture_parser.add_argument("--allow-remote-models", action="store_true")
    e2e_capture_parser.add_argument("--env-file", type=Path)

    e2e_eval_parser = subparsers.add_parser(
        "evaluate-inhouse-calibration",
        help="Evaluate saved in-house calibration captures locally",
    )
    e2e_eval_parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "eval_cases_milestone3_v2_calibration.jsonl",
    )
    e2e_eval_parser.add_argument(
        "--capture",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports"
        / "inhouse_calibration_e2e.jsonl",
    )
    e2e_eval_parser.add_argument("--limit-cases", type=int)
    e2e_eval_parser.add_argument("--output-json", type=Path)

    judge_study_parser = subparsers.add_parser(
        "prepare-judge-study",
        help="Prepare blinded 200/200 human judge annotation sets",
    )
    judge_study_parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "eval_cases_milestone3_v2_calibration.jsonl",
    )
    judge_study_parser.add_argument("--source-results", type=Path, required=True)
    judge_study_parser.add_argument("--output-dir", type=Path, required=True)
    judge_study_parser.add_argument("--seed", default="milestone3-judge-v1")

    judge_status_parser = subparsers.add_parser(
        "judge-study-status",
        help="Report completion of human judge annotation templates",
    )
    judge_status_parser.add_argument("--study-dir", type=Path, required=True)

    judge_review_parser = subparsers.add_parser(
        "review-judge-study",
        help="Run the local blinded human judge review interface",
    )
    judge_review_parser.add_argument("--study-dir", type=Path, required=True)
    judge_review_parser.add_argument(
        "--reviewer",
        choices=("reviewer_a", "reviewer_b"),
        required=True,
    )
    judge_review_parser.add_argument("--port", type=int, default=8765)
    judge_review_parser.add_argument("--section-size", type=int, default=10)
    judge_review_parser.add_argument("--open", action="store_true")
    judge_review_parser.add_argument(
        "--allow-reviewer-switch",
        action="store_true",
        help=(
            "Expose both reviewer workspaces in one trusted local process; "
            "do not use for independent blinded review"
        ),
    )

    judge_recommendation_parser = subparsers.add_parser(
        "prepare-judge-recommendations",
        help="Create separate rubric recommendations for faster human review",
    )
    judge_recommendation_parser.add_argument(
        "--study-dir",
        type=Path,
        required=True,
    )

    judge_reconciliation_ui_parser = subparsers.add_parser(
        "review-judge-reconciliation",
        help="Review three-way labels and adjudicate human disagreements",
    )
    judge_reconciliation_ui_parser.add_argument(
        "--study-dir",
        type=Path,
        required=True,
    )
    judge_reconciliation_ui_parser.add_argument(
        "--port",
        type=int,
        default=8770,
    )
    judge_reconciliation_ui_parser.add_argument(
        "--section-size",
        type=int,
        default=10,
    )
    judge_reconciliation_ui_parser.add_argument(
        "--open",
        action="store_true",
    )

    judge_reconcile_parser = subparsers.add_parser(
        "reconcile-judge-study",
        help="Measure reviewer agreement and prepare adjudication items",
    )
    judge_reconcile_parser.add_argument("--study-dir", type=Path, required=True)

    judge_finalize_parser = subparsers.add_parser(
        "finalize-judge-study",
        help="Compile reviewer consensus and adjudications into human labels",
    )
    judge_finalize_parser.add_argument("--study-dir", type=Path, required=True)

    judge_capture_parser = subparsers.add_parser(
        "capture-judge-study",
        help="Capture resumable in-house judge predictions without human labels",
    )
    judge_capture_parser.add_argument("--study-dir", type=Path, required=True)
    judge_capture_parser.add_argument(
        "--source-cases",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "eval_cases_milestone3_v2_calibration.jsonl",
    )
    judge_capture_parser.add_argument("--source-results", type=Path, required=True)
    judge_capture_parser.add_argument("--judge-model", required=True)
    judge_capture_parser.add_argument("--output", type=Path, required=True)
    judge_capture_parser.add_argument("--manifest", type=Path, required=True)
    judge_capture_parser.add_argument("--max-concurrency", type=int, default=1)
    judge_capture_parser.add_argument("--retry-failures", action="store_true")
    judge_capture_parser.add_argument("--allow-remote-models", action="store_true")
    judge_capture_parser.add_argument("--env-file", type=Path)

    judge_evaluate_parser = subparsers.add_parser(
        "evaluate-judge-study",
        help="Compare saved judge predictions against adjudicated human labels",
    )
    judge_evaluate_parser.add_argument("--study-dir", type=Path, required=True)
    judge_evaluate_parser.add_argument(
        "--predictions",
        type=Path,
        action="append",
        required=True,
    )
    judge_evaluate_parser.add_argument("--output-json", type=Path, required=True)

    holdout_prepare_parser = subparsers.add_parser(
        "prepare-holdout-review",
        help="Prepare separate blinded reviewer files for the frozen holdout",
    )
    holdout_prepare_parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "eval_cases_milestone3_v2_holdout.jsonl",
    )
    holdout_prepare_parser.add_argument("--output-dir", type=Path, required=True)

    holdout_status_parser = subparsers.add_parser(
        "holdout-review-status",
        help="Report independent holdout review completion",
    )
    holdout_status_parser.add_argument("--study-dir", type=Path, required=True)

    holdout_reconcile_parser = subparsers.add_parser(
        "reconcile-holdout-review",
        help="Measure holdout reviewer agreement and prepare disagreements",
    )
    holdout_reconcile_parser.add_argument("--study-dir", type=Path, required=True)

    holdout_finalize_parser = subparsers.add_parser(
        "finalize-holdout-review",
        help="Compile consensus and adjudications into canonical annotations",
    )
    holdout_finalize_parser.add_argument("--study-dir", type=Path, required=True)
    holdout_finalize_parser.add_argument(
        "--output-annotations", type=Path, required=True
    )
    holdout_finalize_parser.add_argument("--replace", action="store_true")

    final_evidence_parser = subparsers.add_parser(
        "build-final-evidence",
        help="Build the final common-split calibration report",
    )
    final_evidence_parser.add_argument(
        "--deterministic-report",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports"
        / "inhouse_common_split_calibration_v3.json",
    )
    final_evidence_parser.add_argument(
        "--model-report",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports"
        / "inhouse_retrieval_recovery_calibration_v4.json",
    )
    final_evidence_parser.add_argument(
        "--failure-report",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports"
        / "inhouse_calibration_failure_analysis_v4.json",
    )
    final_evidence_parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports"
        / "final_calibration_evidence.json",
    )
    final_evidence_parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports"
        / "final_calibration_evidence.md",
    )

    final_freeze_parser = subparsers.add_parser(
        "seal-final-config",
        help="Seal runtime artifacts after holdout annotation is complete",
    )
    final_freeze_parser.add_argument("--dataset-manifest", type=Path, required=True)
    final_freeze_parser.add_argument("--calibration-report", type=Path, required=True)
    final_freeze_parser.add_argument("--policy", type=Path, required=True)
    final_freeze_parser.add_argument("--course-corpus", type=Path, required=True)
    final_freeze_parser.add_argument("--index-manifest", type=Path, required=True)
    final_freeze_parser.add_argument("--output-json", type=Path, required=True)

    final_readiness_parser = subparsers.add_parser(
        "check-final-readiness",
        help="Verify human, judge, calibration, and configuration gates",
    )
    final_readiness_parser.add_argument("--dataset-manifest", type=Path, required=True)
    final_readiness_parser.add_argument("--judge-report", type=Path, required=True)
    final_readiness_parser.add_argument("--selected-judge-model", required=True)
    final_readiness_parser.add_argument("--calibration-report", type=Path, required=True)
    final_readiness_parser.add_argument(
        "--configuration-manifest", type=Path, required=True
    )
    final_readiness_parser.add_argument("--output-json", type=Path)

    index_parser = subparsers.add_parser("build-index", help="Build a local Chroma vector index")
    _add_profile_arg(index_parser)
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
    _add_profile_arg(model_config_parser)
    model_config_parser.add_argument("--env-file", type=Path)

    course_parser = subparsers.add_parser("normalize-course-corpus", help="Normalize markdown course corpus to JSONL")
    course_parser.add_argument("--source", type=Path, default=default_course_source_path())
    course_parser.add_argument("--output", type=Path, default=default_course_output_path())
    course_parser.add_argument("--course-id", default="python-intro")

    visualize_parser = subparsers.add_parser("visualize", help="Write a static HTML RAG pipeline visualization")
    _add_profile_arg(visualize_parser)
    visualize_parser.add_argument("--corpus", dest="command_corpus", type=Path)
    visualize_parser.add_argument("--index-dir", type=Path, default=default_index_path())
    visualize_parser.add_argument("--course-id", default="guardrails-101")
    visualize_parser.add_argument("--mode", choices=["baseline", "guardrailed"], default="guardrailed")
    visualize_parser.add_argument("--retriever", choices=["lexical", "langchain", "vector"], default="lexical")
    _add_embedding_args(visualize_parser)
    _add_generation_args(visualize_parser)
    _add_guard_classifier_args(visualize_parser)
    _add_grounding_args(visualize_parser)
    visualize_parser.add_argument("--policy", type=Path)
    _add_guard_embedding_args(visualize_parser)
    visualize_parser.add_argument("--question", required=True)
    visualize_parser.add_argument("--output", type=Path, required=True)

    workshop3_demo_parser = subparsers.add_parser(
        "workshop3-demo",
        help="Write the Workshop 3 guardrail comparison demo",
    )
    workshop3_demo_parser.add_argument(
        "--evidence",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports"
        / "final_calibration_evidence.json",
    )
    workshop3_demo_parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports"
        / "workshop3_guardrail_demo.html",
    )
    workshop3_demo_parser.add_argument("--live", action="store_true")
    workshop3_demo_parser.add_argument("--allow-remote-models", action="store_true")
    workshop3_demo_parser.add_argument("--env-file", type=Path)
    workshop3_demo_parser.add_argument("--open", action="store_true")

    args = parser.parse_args()
    try:
        apply_model_profile(args)
    except (InHouseEndpointError, ValueError) as exc:
        parser.error(str(exc))
    corpus_path = getattr(args, "command_corpus", None) or args.corpus

    if args.command == "workshop3-demo":
        try:
            result = write_workshop3_demo(
                evidence_path=args.evidence,
                output_path=args.output,
                live=args.live,
                allow_remote_models=args.allow_remote_models,
                env_file=args.env_file,
                open_browser=args.open,
            )
        except (
            OSError,
            ValueError,
            VectorIndexError,
            InHouseEndpointError,
            RemoteModelsNotAllowedError,
            MissingModelCredentialError,
            RemoteModelCallError,
        ) as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2))
        return

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
        if args.profile == "local":
            print(json.dumps(openai_config_summary(args.env_file), indent=2))
        else:
            print(json.dumps(model_profile_summary(args.profile, args.env_file), indent=2))
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

    if args.command == "capture-model-calibration":
        try:
            manifest = run_model_calibration_capture(
                component=args.component,
                config=OpenAIModelConfig(
                    classifier_model=(
                        args.classifier_model or DEFAULT_OPENAI_CLASSIFIER_MODEL
                    ),
                    judge_model=args.judge_model or DEFAULT_OPENAI_JUDGE_MODEL,
                    allow_remote_models=args.allow_remote_models,
                    env_file=args.env_file,
                ),
                classifier_cases_path=args.classifier_cases,
                judge_cases_path=args.judge_cases,
                source_cases_path=args.source_cases,
                source_results_path=args.source_results,
                classifier_output_path=args.classifier_output,
                judge_output_path=args.judge_output,
                manifest_output_path=args.manifest_output,
                limit_cases=args.limit_cases,
                selection_strategy=args.selection_strategy,
            )
        except (
            OSError,
            ValueError,
            RemoteModelsNotAllowedError,
            MissingModelCredentialError,
        ) as exc:
            parser.error(str(exc))
        print(json.dumps(manifest, indent=2))
        return

    if args.command == "capture-v2-classifier":
        try:
            manifest = run_v2_classifier_capture(
                config=OpenAIModelConfig(
                    classifier_model=args.classifier_model or DEFAULT_OPENAI_CLASSIFIER_MODEL,
                    allow_remote_models=args.allow_remote_models,
                    env_file=args.env_file,
                ),
                development_cases_path=args.development_cases,
                calibration_cases_path=args.calibration_cases,
                corpus_path=args.course_corpus,
                policy_path=args.policy,
                output_path=args.output,
                manifest_path=args.manifest,
                limit_cases=args.limit_cases,
                max_concurrency=args.max_concurrency,
                retry_failures=args.retry_failures,
            )
        except (
            OSError,
            ValueError,
            RemoteModelsNotAllowedError,
            MissingModelCredentialError,
        ) as exc:
            parser.error(str(exc))
        print(json.dumps(manifest, indent=2))
        return

    if args.command == "evaluate-v2-classifier":
        try:
            report = evaluate_v2_classifier_capture(
                development_cases_path=args.development_cases,
                calibration_cases_path=args.calibration_cases,
                predictions_path=args.predictions,
                limit_cases=args.limit_cases,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(report, indent=2))
        return

    if args.command == "prepare-inhouse-bge":
        try:
            manifest = prepare_inhouse_bge(
                config=OpenAIModelConfig(
                    embedding_model=args.embedding_model,
                    allow_remote_models=args.allow_remote_models,
                    env_file=args.env_file,
                ),
                development_cases_path=args.development_cases,
                calibration_cases_path=args.calibration_cases,
                corpus_path=args.course_corpus,
                policy_path=args.policy,
                index_dir=args.index_dir,
                cache_path=args.embedding_cache,
                manifest_path=args.manifest,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        except (
            OSError,
            ValueError,
            VectorIndexError,
            RemoteModelsNotAllowedError,
            MissingModelCredentialError,
            RemoteModelCallError,
        ) as exc:
            parser.error(str(exc))
        print(json.dumps(manifest, indent=2))
        return

    if args.command == "calibrate-inhouse-bge":
        try:
            summary, details = run_bge_common_split_evaluation(
                config=OpenAIModelConfig(
                    embedding_model=args.embedding_model,
                    allow_remote_models=args.allow_remote_models,
                    env_file=args.env_file,
                ),
                development_cases_path=args.development_cases,
                calibration_cases_path=args.calibration_cases,
                corpus_path=args.course_corpus,
                policy_path=args.policy,
                bge_index_dir=args.bge_index_dir,
                hashing_index_dir=args.hashing_index_dir,
                cache_path=args.embedding_cache,
                course_id=args.course_id,
                top_k=args.top_k,
            )
        except (
            OSError,
            ValueError,
            VectorIndexError,
            RemoteModelsNotAllowedError,
            MissingModelCredentialError,
            RemoteModelCallError,
        ) as exc:
            parser.error(str(exc))
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        args.output_details_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_details_json.write_text(
            json.dumps(details, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "capture-inhouse-calibration":
        try:
            manifest = run_calibration_e2e_capture(
                config=OpenAIModelConfig(
                    embedding_model=args.embedding_model,
                    answer_model=args.answer_model,
                    classifier_model=args.classifier_model,
                    entailment_model=args.entailment_model,
                    allow_remote_models=args.allow_remote_models,
                    env_file=args.env_file,
                ),
                calibration_cases_path=args.cases,
                corpus_path=args.course_corpus,
                policy_path=args.policy,
                index_dir=args.index_dir,
                cache_path=args.embedding_cache,
                output_path=args.output,
                manifest_path=args.manifest,
                evidence_min_score=args.evidence_min_score,
                policy_context_top_k=args.policy_context_top_k,
                policy_context_min_score=args.policy_context_min_score,
                entailment_min_confidence=args.entailment_min_confidence,
                course_id=args.course_id,
                limit_cases=args.limit_cases,
                case_ids=args.case_ids,
                max_concurrency=args.max_concurrency,
                retry_failures=args.retry_failures,
            )
        except (
            OSError,
            ValueError,
            VectorIndexError,
            RemoteModelsNotAllowedError,
            MissingModelCredentialError,
            RemoteModelCallError,
        ) as exc:
            parser.error(str(exc))
        print(json.dumps(manifest, indent=2))
        return

    if args.command == "evaluate-inhouse-calibration":
        try:
            report = evaluate_calibration_e2e_capture(
                calibration_cases_path=args.cases,
                output_path=args.capture,
                limit_cases=args.limit_cases,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(report, indent=2))
        return

    if args.command == "prepare-judge-study":
        try:
            manifest = prepare_judge_study(
                cases_path=args.cases,
                source_results_path=args.source_results,
                output_dir=args.output_dir,
                seed=args.seed,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(manifest, indent=2))
        return

    if args.command == "judge-study-status":
        try:
            status = _judge_study_status(args.study_dir)
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(status, indent=2))
        return

    if args.command == "review-judge-study":
        try:
            serve_review_ui(
                study_dir=args.study_dir,
                reviewer=args.reviewer,
                port=args.port,
                section_size=args.section_size,
                open_browser=args.open,
                allow_reviewer_switch=args.allow_reviewer_switch,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        return

    if args.command == "prepare-judge-recommendations":
        try:
            report = prepare_review_recommendations(args.study_dir)
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(report, indent=2))
        return

    if args.command == "review-judge-reconciliation":
        try:
            serve_reconciliation_ui(
                study_dir=args.study_dir,
                port=args.port,
                section_size=args.section_size,
                open_browser=args.open,
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        return

    if args.command == "reconcile-judge-study":
        try:
            report = reconcile_human_annotations(
                items_paths=[
                    args.study_dir / f"{split}_items.jsonl"
                    for split in JUDGE_SPLITS
                ],
                reviewer_a_paths=[
                    args.study_dir / f"{split}_reviewer_a.jsonl"
                    for split in JUDGE_SPLITS
                ],
                reviewer_b_paths=[
                    args.study_dir / f"{split}_reviewer_b.jsonl"
                    for split in JUDGE_SPLITS
                ],
                disagreements_output=args.study_dir / "judge_disagreements.jsonl",
                report_output=args.study_dir / "judge_human_agreement.json",
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(report, indent=2))
        return

    if args.command == "finalize-judge-study":
        try:
            report = finalize_human_ground_truth(study_dir=args.study_dir)
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(report, indent=2))
        return

    if args.command == "capture-judge-study":
        try:
            manifest = run_judge_study_capture(
                config=OpenAIModelConfig(
                    judge_model=args.judge_model,
                    allow_remote_models=args.allow_remote_models,
                    env_file=args.env_file,
                ),
                study_dir=args.study_dir,
                source_cases_path=args.source_cases,
                source_results_path=args.source_results,
                output_path=args.output,
                manifest_path=args.manifest,
                max_concurrency=args.max_concurrency,
                retry_failures=args.retry_failures,
            )
        except (
            OSError,
            ValueError,
            InHouseEndpointError,
            RemoteModelsNotAllowedError,
            MissingModelCredentialError,
        ) as exc:
            parser.error(str(exc))
        print(json.dumps(manifest, indent=2))
        return

    if args.command == "evaluate-judge-study":
        try:
            report = evaluate_judge_study_models(
                study_dir=args.study_dir,
                prediction_paths=args.predictions,
                output_path=args.output_json,
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(report, indent=2))
        return

    if args.command == "prepare-holdout-review":
        try:
            report = prepare_holdout_review(
                cases_path=args.cases,
                output_dir=args.output_dir,
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(report, indent=2))
        return

    if args.command == "holdout-review-status":
        try:
            report = holdout_review_status(args.study_dir)
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(report, indent=2))
        return

    if args.command == "reconcile-holdout-review":
        try:
            report = reconcile_holdout_review(args.study_dir)
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(report, indent=2))
        return

    if args.command == "finalize-holdout-review":
        try:
            report = finalize_holdout_review(
                study_dir=args.study_dir,
                output_path=args.output_annotations,
                replace=args.replace,
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(report, indent=2))
        return

    if args.command == "build-final-evidence":
        try:
            report = write_calibration_evidence(
                deterministic_path=args.deterministic_report,
                model_path=args.model_report,
                failure_path=args.failure_report,
                output_json=args.output_json,
                output_markdown=args.output_markdown,
            )
        except FinalEvidenceError as exc:
            parser.error(str(exc))
        print(json.dumps(report, indent=2))
        return

    if args.command == "seal-final-config":
        try:
            report = seal_runtime_configuration(
                dataset_manifest_path=args.dataset_manifest,
                calibration_report_path=args.calibration_report,
                policy_path=args.policy,
                corpus_path=args.course_corpus,
                index_manifest_path=args.index_manifest,
                output_path=args.output_json,
            )
        except FinalEvidenceError as exc:
            parser.error(str(exc))
        print(json.dumps(report, indent=2))
        return

    if args.command == "check-final-readiness":
        try:
            report = assess_final_readiness_from_files(
                dataset_manifest_path=args.dataset_manifest,
                judge_report_path=args.judge_report,
                selected_judge_model=args.selected_judge_model,
                calibration_report_path=args.calibration_report,
                configuration_manifest_path=args.configuration_manifest,
                output_path=args.output_json,
            )
        except FinalEvidenceError as exc:
            parser.error(str(exc))
        print(json.dumps(report, indent=2))
        if not report["ready"]:
            raise SystemExit(1)
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
                embedding_cache_path=args.embedding_cache,
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
                embedding_cache_path=args.embedding_cache,
                generator=args.generator,
                answer_model=args.answer_model,
                guard_classifier=args.guard_classifier,
                classifier_model=args.classifier_model,
                classifier_strategy=args.classifier_strategy,
                retrieval_top_k=args.retrieval_top_k,
                evidence_min_score=args.evidence_min_score,
                policy_context_top_k=args.policy_context_top_k,
                policy_context_min_score=args.policy_context_min_score,
                entailment_verifier=args.entailment_verifier,
                entailment_model=args.entailment_model,
                entailment_min_confidence=args.entailment_min_confidence,
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
        cases = _load_run_cases(parser, args.cases, corpus_path)
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
                    embedding_cache_path=args.embedding_cache,
                    generator=args.generator,
                    answer_model=args.answer_model,
                    guard_classifier=classifier,
                    classifier_model=args.classifier_model,
                    classifier_strategy=profile.get(
                        "classifier_strategy",
                        "ambiguous",
                    ),
                    retrieval_top_k=args.retrieval_top_k,
                    evidence_min_score=args.evidence_min_score,
                    policy_context_top_k=args.policy_context_top_k,
                    policy_context_min_score=args.policy_context_min_score,
                    entailment_verifier=args.entailment_verifier,
                    entailment_model=args.entailment_model,
                    entailment_min_confidence=args.entailment_min_confidence,
                    retrieval_embedder=retrieval_embedder,
                )
                comparison_results = run_evaluation(comparison_assistant, cases)
                comparison_details[label] = [
                    asdict(result) for result in comparison_results
                ]
                comparison_summary = profile | summarize(comparison_results)
                comparison_summary["confidence_intervals"] = (
                    bootstrap_confidence_intervals(comparison_results)
                )
                comparison_summary["eval_split"] = args.case_split
                preloads = {}
                if retrieval_preload:
                    preloads["retrieval"] = retrieval_preload
                if guard_preload and label in {
                    "similarity_plus_shared_controls",
                    "hybrid_policy_guardrails",
                    "complete_inhouse_hybrid",
                }:
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
            embedding_cache_path=getattr(args, "embedding_cache", None),
            generator=getattr(args, "generator", "extractive"),
            answer_model=getattr(args, "answer_model", None),
            guard_classifier=getattr(args, "guard_classifier", "none"),
            classifier_model=getattr(args, "classifier_model", None),
            classifier_strategy=getattr(args, "classifier_strategy", "ambiguous"),
            retrieval_top_k=getattr(args, "retrieval_top_k", 3),
            evidence_min_score=getattr(args, "evidence_min_score", None),
            policy_context_top_k=getattr(args, "policy_context_top_k", 0),
            policy_context_min_score=getattr(
                args,
                "policy_context_min_score",
                None,
            ),
            entailment_verifier=getattr(args, "entailment_verifier", "none"),
            entailment_model=getattr(args, "entailment_model", None),
            entailment_min_confidence=getattr(
                args,
                "entailment_min_confidence",
                0.80,
            ),
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

    cases = _load_run_cases(parser, args.cases, corpus_path)
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
    parser.add_argument("--embedding-cache", type=Path)


def _add_profile_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=MODEL_PROFILES, default="local")


def _load_run_cases(
    parser: argparse.ArgumentParser,
    cases_path: Path,
    corpus_path: Path,
) -> list[EvalCase]:
    try:
        return load_evaluation_cases_for_run(
            cases_path,
            corpus_path=corpus_path,
        )
    except DatasetValidationError as exc:
        parser.error(str(exc))


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generator", choices=["extractive", "openai"], default="extractive")
    parser.add_argument("--answer-model")


def _add_guard_classifier_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--guard-classifier", choices=["none", "openai"], default="none")
    parser.add_argument("--classifier-model")
    parser.add_argument(
        "--classifier-strategy",
        choices=["ambiguous", "always"],
        default="ambiguous",
    )


def _add_grounding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument("--evidence-min-score", type=_finite_float)
    parser.add_argument("--policy-context-top-k", type=int, default=0)
    parser.add_argument("--policy-context-min-score", type=_finite_float)
    parser.add_argument(
        "--entailment-verifier",
        choices=["none", "openai"],
        default="none",
    )
    parser.add_argument("--entailment-model")
    parser.add_argument(
        "--entailment-min-confidence",
        type=_unit_float,
        default=0.80,
    )


def _add_guard_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--guard-embedding-provider", choices=["hashing", "openai"], default="hashing")
    parser.add_argument("--guard-embedding-model")


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


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
        cache_path=getattr(args, "embedding_cache", None),
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
            cache_path=getattr(args, "embedding_cache", None),
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
            cache_path=getattr(args, "embedding_cache", None),
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
    similarity_policy = replace(
        policy,
        input_rules=(),
        input_fuzzy_rules=(),
        output_rules=(),
        output_fuzzy_rules=(),
        context_rules=(),
        context_fuzzy_rules=(),
    )
    classifier_only_policy = replace(
        policy,
        input_rules=(),
        input_similarity_rules=(),
        input_fuzzy_rules=(),
        output_rules=(),
        output_fuzzy_rules=(),
        context_rules=(),
        context_fuzzy_rules=(),
    )
    shared_controls = ["metadata_filter", "citation_requirement"]
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
                "shared_controls": [],
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
                "shared_controls": shared_controls,
            },
        ),
        (
            "fuzzy_plus_shared_controls",
            "guardrailed",
            fuzzy_policy,
            "none",
            {
                "technique": "fuzzy rules with shared metadata and citation controls",
                "guardrail_layers": [
                    "text_normalization",
                    "fuzzy_input",
                    "context_fuzzy_sanitization",
                    "output_fuzzy_check",
                ],
                "latency_expected": "low-medium",
                "robustness_expected": "medium_on_typos_and_near_matches",
                "implementation_effort": "medium",
                "shared_controls": shared_controls,
            },
        ),
        (
            "similarity_plus_shared_controls",
            "guardrailed",
            similarity_policy,
            "none",
            {
                "technique": (
                    "embedding similarity rules with shared metadata and "
                    "citation controls"
                ),
                "guardrail_layers": ["embedding_similarity_input"],
                "latency_expected": "provider-dependent",
                "robustness_expected": "medium_on_semantic_variants",
                "implementation_effort": "medium-high",
                "shared_controls": shared_controls,
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
                "shared_controls": shared_controls,
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
                "shared_controls": shared_controls,
            },
        ),
    ]
    if args.guard_classifier != "none":
        scenarios.extend(
            [
            (
                "qwen_classifier_only",
                "guardrailed",
                classifier_only_policy,
                args.guard_classifier,
                {
                    "technique": "Qwen classifier with shared metadata and citation controls",
                    "guardrail_layers": [
                        "model_classifier",
                        "metadata_filter",
                        "citation_requirement",
                    ],
                    "classifier_strategy": "always",
                    "latency_expected": "high",
                    "robustness_expected": "model-dependent",
                    "implementation_effort": "high",
                    "shared_controls": shared_controls,
                },
            ),
            (
                "complete_inhouse_hybrid",
                "guardrailed",
                policy,
                args.guard_classifier,
                {
                    "technique": "hybrid policy + model classifier for unresolved prompts",
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
                    "shared_controls": shared_controls,
                    "classifier_strategy": "always",
                },
            ),
            ]
        )
    return scenarios


def _judge_study_status(study_dir: Path) -> dict[str, object]:
    summaries: dict[str, object] = {}
    for split in JUDGE_SPLITS:
        items_path = study_dir / f"{split}_items.jsonl"
        item_ids = {
            str(json.loads(line)["item_id"])
            for line in items_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        for reviewer in ("reviewer_a", "reviewer_b"):
            _annotations, summary = validate_annotation_file(
                study_dir / f"{split}_{reviewer}.jsonl",
                expected_item_ids=item_ids,
                complete=False,
            )
            summaries[f"{split}:{reviewer}"] = summary
    return {
        "study_dir": str(study_dir),
        "human_ground_truth_ready": all(
            bool(summary["complete"])
            for summary in summaries.values()
        ),
        "annotation_files": summaries,
    }


if __name__ == "__main__":
    main()
