from __future__ import annotations

from hashlib import sha256
from math import ceil
from pathlib import Path

from .dispositions import ResponseDisposition
from .embeddings import HashingEmbedder, TextEmbedder, cosine_similarity, create_embedder
from .evaluation_dataset import DEFAULT_DATASET_MANIFEST_PATH
from .evaluation import EvalCase, load_eval_cases
from .guard_text import normalize_guard_text
from .guardrail_policy import load_guardrail_policy
from .inhouse_experiment import (
    _verify_development_and_calibration_dataset,
    derive_classifier_label,
)
from .model_config import (
    OpenAIModelConfig,
    ensure_openai_api_key,
    ensure_remote_models_allowed,
    openai_request_policy,
)
from .model_profiles import ensure_inhouse_endpoint
from .retrieval_routing import route_retrieval_query
from .vector import DEFAULT_VECTOR_MIN_SCORE, VectorRetriever, build_vector_index


DOCUMENT_RECALL_TARGET = 0.95


def run_bge_common_split_evaluation(
    *,
    config: OpenAIModelConfig,
    development_cases_path: Path,
    calibration_cases_path: Path,
    corpus_path: Path,
    policy_path: Path,
    bge_index_dir: Path,
    hashing_index_dir: Path,
    cache_path: Path,
    course_id: str = "python-intro",
    top_k: int = 3,
    bge_embedder: TextEmbedder | None = None,
    dataset_manifest_path: Path = DEFAULT_DATASET_MANIFEST_PATH,
) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    ensure_remote_models_allowed(config)
    ensure_openai_api_key(config)
    endpoint_host = ensure_inhouse_endpoint(config.env_file)
    development = load_eval_cases(development_cases_path)
    calibration = load_eval_cases(calibration_cases_path)
    _require_split(development, "development")
    _require_split(calibration, "calibration")
    dataset_evidence = _verify_development_and_calibration_dataset(
        dataset_manifest_path,
        development_cases_path=development_cases_path,
        calibration_cases_path=calibration_cases_path,
    )
    cases = development + calibration

    bge = bge_embedder or create_embedder(
        "openai",
        model=config.embedding_model,
        model_config=config,
        allow_remote_models=config.allow_remote_models,
        env_file=config.env_file,
        cache_path=cache_path,
    )
    if bge.model_name != config.embedding_model:
        raise ValueError("BGE embedder model does not match the configured model")
    hashing = HashingEmbedder()
    build_vector_index(
        corpus_path,
        hashing_index_dir,
        embedding_provider="hashing",
        embedder=hashing,
    )
    retrievers = {
        "bge_m3": VectorRetriever(
            bge_index_dir,
            min_score=DEFAULT_VECTOR_MIN_SCORE,
            embedding_provider="openai",
            embedding_model=config.embedding_model,
            embedder=bge,
            corpus_path=corpus_path,
        ),
        "hashing": VectorRetriever(
            hashing_index_dir,
            min_score=DEFAULT_VECTOR_MIN_SCORE,
            embedding_provider="hashing",
            embedder=hashing,
            corpus_path=corpus_path,
        ),
    }
    embedders = {"bge_m3": bge, "hashing": hashing}
    summaries: dict[str, object] = {}
    details: dict[str, list[dict[str, object]]] = {}
    for name in ("hashing", "bge_m3"):
        rows = _score_cases(
            cases,
            retriever=retrievers[name],
            embedder=embedders[name],
            policy_path=policy_path,
            course_id=course_id,
            top_k=top_k,
        )
        details[name] = rows
        summaries[name] = _summarize_technique(rows)

    summary = {
        "evidence_scope": "common_split_embedding_and_retrieval_calibration",
        "endpoint_host": endpoint_host,
        "request_policy": openai_request_policy(config),
        "dataset_version": dataset_evidence["dataset_version"],
        "dataset_manifest_sha256": dataset_evidence["dataset_manifest_sha256"],
        "models": {
            "bge_m3": config.embedding_model,
            "hashing": hashing.model_name,
        },
        "case_counts": {
            "development": len(development),
            "calibration": len(calibration),
        },
        "threshold_selection_split": "development",
        "threshold_validation_split": "calibration",
        "holdout_used": False,
        "top_k": top_k,
        "runtime_retriever_min_score": DEFAULT_VECTOR_MIN_SCORE,
        "corpus_sha256": _file_sha256(corpus_path),
        "split_sha256": {
            "development": _file_sha256(development_cases_path),
            "calibration": _file_sha256(calibration_cases_path),
        },
        "techniques": summaries,
    }
    return summary, details


def _score_cases(
    cases: list[EvalCase],
    *,
    retriever: VectorRetriever,
    embedder: TextEmbedder,
    policy_path: Path,
    course_id: str,
    top_k: int,
) -> list[dict[str, object]]:
    policy = load_guardrail_policy(policy_path, similarity_embedder=embedder)
    rule_vectors = {
        rule.trigger: embedder.embed_many(
            [normalize_guard_text(example) for example in rule.examples]
        )
        for rule in policy.input_similarity_rules
    }
    normalized = [normalize_guard_text(case.question) for case in cases]
    query_vectors = embedder.embed_many(normalized)
    rows = []
    for case, query_vector in zip(cases, query_vectors, strict=True):
        classifier_label = derive_classifier_label(case)
        retrieval_attempted = (
            case.resolved_expected_behavior() is not ResponseDisposition.BLOCK
        )
        retrieval_query = (
            route_retrieval_query(case.question, {classifier_label})
            if retrieval_attempted
            else None
        )
        matches = (
            retriever.search(
                retrieval_query,
                course_id=course_id,
                allowed_visibility=set(policy.allowed_visibility),
                top_k=retriever.indexed_chunks,
            )
            if retrieval_query is not None
            else []
        )
        runtime_matches = matches[:top_k]
        retrieved_doc_ids = list(
            dict.fromkeys(chunk.doc_id for chunk, _score in runtime_matches)
        )
        ranked_doc_ids = list(dict.fromkeys(chunk.doc_id for chunk, _score in matches))[
            :top_k
        ]
        rows.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "family_id": case.family_id,
                "language": case.language,
                "expected_classifier_label": classifier_label,
                "retrieval_attempted": retrieval_attempted,
                "retrieval_query": retrieval_query,
                "evidence_available": case.evidence_available,
                "expected_doc_ids": list(case.expected_doc_ids or []),
                "retrieved_doc_ids": retrieved_doc_ids,
                "ranked_doc_ids": ranked_doc_ids,
                "retrieved_chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "score": float(score),
                    }
                    for chunk, score in runtime_matches
                ],
                "document_ranking_candidate_limit": (
                    retriever.indexed_chunks if retrieval_attempted else 0
                ),
                "retrieval_scores": [
                    float(score) for _chunk, score in runtime_matches
                ],
                "top_retrieval_score": (
                    float(matches[0][1]) if matches else -1.0
                ),
                "guard_similarity_scores": {
                    rule.trigger: max(
                        cosine_similarity(query_vector, example_vector)
                        for example_vector in rule_vectors[rule.trigger]
                    )
                    for rule in policy.input_similarity_rules
                },
            }
        )
    return rows


def _summarize_technique(rows: list[dict[str, object]]) -> dict[str, object]:
    development = [row for row in rows if row["split"] == "development"]
    calibration = [row for row in rows if row["split"] == "calibration"]
    retrieval_development = [
        row
        for row in development
        if row["retrieval_attempted"] is True
    ]
    retrieval_calibration = [
        row
        for row in calibration
        if row["retrieval_attempted"] is True
    ]
    retrieval_threshold = _select_threshold(
        [float(row["top_retrieval_score"]) for row in retrieval_development],
        [bool(row["evidence_available"]) for row in retrieval_development],
    )
    document_recall_threshold = _select_document_recall_threshold(
        retrieval_development,
        target_recall=DOCUMENT_RECALL_TARGET,
    )
    guard_triggers = tuple(development[0]["guard_similarity_scores"]) if development else ()
    guard_summary = {}
    for trigger in guard_triggers:
        threshold = _select_threshold(
            [float(row["guard_similarity_scores"][trigger]) for row in development],
            [row["expected_classifier_label"] == trigger for row in development],
        )
        guard_summary[trigger] = {
            "selected_threshold": threshold,
            "development": _binary_metrics(
                [float(row["guard_similarity_scores"][trigger]) for row in development],
                [row["expected_classifier_label"] == trigger for row in development],
                threshold,
            ),
            "calibration": _binary_metrics(
                [float(row["guard_similarity_scores"][trigger]) for row in calibration],
                [row["expected_classifier_label"] == trigger for row in calibration],
                threshold,
            ),
        }
    return {
        "retrieval": {
            "selected_threshold": retrieval_threshold,
            "document_recall_target": DOCUMENT_RECALL_TARGET,
            "document_recall_threshold": document_recall_threshold,
            "development": _retrieval_metrics(
                retrieval_development,
                retrieval_threshold,
            ),
            "calibration": _retrieval_metrics(
                retrieval_calibration,
                retrieval_threshold,
            ),
            "development_at_document_recall_threshold": (
                _retrieval_metrics(retrieval_development, document_recall_threshold)
                if document_recall_threshold is not None
                else None
            ),
            "calibration_at_document_recall_threshold": (
                _retrieval_metrics(retrieval_calibration, document_recall_threshold)
                if document_recall_threshold is not None
                else None
            ),
        },
        "guard_similarity": guard_summary,
    }


def _retrieval_metrics(
    rows: list[dict[str, object]],
    threshold: float,
) -> dict[str, int | float]:
    scores = [float(row["top_retrieval_score"]) for row in rows]
    targets = [bool(row["evidence_available"]) for row in rows]
    metrics = _binary_metrics(scores, targets, threshold)
    evaluable = [
        row
        for row in rows
        if row["evidence_available"] is True and row["expected_doc_ids"]
    ]
    chunk_budget_recalls = [
        len(set(row["retrieved_doc_ids"]) & set(row["expected_doc_ids"]))
        / len(set(row["expected_doc_ids"]))
        for row in evaluable
    ]
    document_recalls = [
        len(set(row["ranked_doc_ids"]) & set(row["expected_doc_ids"]))
        / len(set(row["expected_doc_ids"]))
        for row in evaluable
    ]
    thresholded_recalls = [
        len(
            {
                str(chunk["doc_id"])
                for chunk in row.get("retrieved_chunks", [])
                if float(chunk["score"]) >= threshold
            }
            & set(row["expected_doc_ids"])
        )
        / len(set(row["expected_doc_ids"]))
        for row in evaluable
    ]
    metrics["retrieval_evaluable_cases"] = len(evaluable)
    metrics["document_recall_within_top_k_chunks"] = (
        round(sum(chunk_budget_recalls) / len(chunk_budget_recalls), 4)
        if chunk_budget_recalls
        else 0.0
    )
    metrics["document_hit_rate_within_top_k_chunks"] = (
        round(
            sum(recall > 0 for recall in chunk_budget_recalls)
            / len(chunk_budget_recalls),
            4,
        )
        if chunk_budget_recalls
        else 0.0
    )
    metrics["document_recall_after_evidence_threshold"] = (
        round(sum(thresholded_recalls) / len(thresholded_recalls), 4)
        if thresholded_recalls
        else 0.0
    )
    metrics["document_hit_rate_after_evidence_threshold"] = (
        round(
            sum(recall > 0 for recall in thresholded_recalls)
            / len(thresholded_recalls),
            4,
        )
        if thresholded_recalls
        else 0.0
    )
    metrics["document_recall_at_k"] = (
        round(sum(document_recalls) / len(document_recalls), 4)
        if document_recalls
        else 0.0
    )
    metrics["document_hit_rate_at_k"] = (
        round(sum(recall > 0 for recall in document_recalls) / len(document_recalls), 4)
        if document_recalls
        else 0.0
    )
    # Backward-compatible aliases retain the old top-k-chunk interpretation.
    metrics["recall_at_k"] = metrics["document_recall_within_top_k_chunks"]
    metrics["hit_rate_at_k"] = metrics["document_hit_rate_within_top_k_chunks"]
    return metrics


def _select_threshold(scores: list[float], targets: list[bool]) -> float:
    if not scores or len(scores) != len(targets):
        raise ValueError("threshold selection requires aligned non-empty scores and targets")
    ordered = sorted(set(scores))
    candidates = [ordered[0] - 1e-9, ordered[-1] + 1e-9]
    candidates.extend((left + right) / 2 for left, right in zip(ordered, ordered[1:]))
    ranked = []
    for threshold in candidates:
        metrics = _binary_metrics(
            scores,
            targets,
            threshold,
            round_metrics=False,
        )
        ranked.append(
            (
                float(metrics["macro_f1"]),
                float(metrics["precision"]),
                -float(metrics["false_positive_rate"]),
                threshold,
            )
        )
    return max(ranked)[-1]


def _select_document_recall_threshold(
    rows: list[dict[str, object]],
    *,
    target_recall: float,
) -> float | None:
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target document recall must be greater than zero and at most one")
    evaluable = [
        row
        for row in rows
        if row["evidence_available"] is True and row["expected_doc_ids"]
    ]
    if not evaluable:
        raise ValueError("document threshold selection requires evidence-labelled rows")
    required_hits = ceil(target_recall * len(evaluable))
    expected_scores: list[float | None] = []
    for row in evaluable:
        expected_doc_ids = set(row["expected_doc_ids"])
        scores = [
            float(chunk["score"])
            for chunk in row.get("retrieved_chunks", [])
            if chunk["doc_id"] in expected_doc_ids
        ]
        expected_scores.append(max(scores) if scores else None)
    available = sorted(
        (score for score in expected_scores if score is not None),
        reverse=True,
    )
    if len(available) < required_hits:
        return None
    return available[required_hits - 1]


def _binary_metrics(
    scores: list[float],
    targets: list[bool],
    threshold: float,
    *,
    round_metrics: bool = True,
) -> dict[str, int | float]:
    predictions = [score >= threshold for score in scores]
    tp = sum(prediction and target for prediction, target in zip(predictions, targets, strict=True))
    fp = sum(prediction and not target for prediction, target in zip(predictions, targets, strict=True))
    tn = sum(not prediction and not target for prediction, target in zip(predictions, targets, strict=True))
    fn = sum(not prediction and target for prediction, target in zip(predictions, targets, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    positive_f1 = _f1(precision, recall)
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    negative_recall = tn / (tn + fp) if tn + fp else 0.0
    negative_f1 = _f1(negative_precision, negative_recall)
    return {
        "total": len(scores),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": _reported_metric(
            (tp + tn) / len(scores) if scores else 0.0,
            round_metrics=round_metrics,
        ),
        "precision": _reported_metric(precision, round_metrics=round_metrics),
        "recall": _reported_metric(recall, round_metrics=round_metrics),
        "false_positive_rate": _reported_metric(
            fp / (fp + tn) if fp + tn else 0.0,
            round_metrics=round_metrics,
        ),
        "macro_f1": _reported_metric(
            (positive_f1 + negative_f1) / 2,
            round_metrics=round_metrics,
        ),
    }


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _reported_metric(value: float, *, round_metrics: bool) -> float:
    return round(value, 4) if round_metrics else value


def _require_split(cases: list[EvalCase], expected: str) -> None:
    if any(case.split != expected for case in cases):
        raise ValueError(f"{expected} dataset contains cases with a different split")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
