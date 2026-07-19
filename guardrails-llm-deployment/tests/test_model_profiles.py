from argparse import Namespace
from pathlib import Path

import pytest

from guardrails_llm.model_profiles import (
    INHOUSE_EMBEDDING_MODEL,
    INHOUSE_EVIDENCE_MIN_SCORE,
    INHOUSE_LLM_MODEL,
    InHouseEndpointError,
    apply_model_profile,
    model_profile_summary,
)


def _runtime_args() -> Namespace:
    return Namespace(
        profile="inhouse",
        command="query",
        env_file=None,
        command_corpus=None,
        index_dir=Path("indexes/chroma"),
        course_id="guardrails-101",
        retriever="lexical",
        embedding_provider="hashing",
        embedding_model=None,
        embedding_cache=None,
        generator="extractive",
        answer_model=None,
        guard_classifier="none",
        classifier_model=None,
        entailment_verifier="none",
        entailment_model=None,
        evidence_min_score=None,
        guard_embedding_provider="hashing",
        guard_embedding_model=None,
        policy=None,
    )


def test_inhouse_profile_selects_bge_and_qwen(monkeypatch) -> None:
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://learning-services4.fokus.fraunhofer.de/litellm/v1",
    )
    args = _runtime_args()

    apply_model_profile(args)

    assert args.retriever == "vector"
    assert args.command_corpus.name == "python_course_docs.jsonl"
    assert args.index_dir.name == "python-course-bge-m3"
    assert args.course_id == "python-intro"
    assert args.embedding_provider == "openai"
    assert args.embedding_model == INHOUSE_EMBEDDING_MODEL
    assert args.embedding_cache.name == "bge-m3.jsonl"
    assert args.guard_embedding_provider == "openai"
    assert args.guard_embedding_model == INHOUSE_EMBEDDING_MODEL
    assert args.generator == "openai"
    assert args.answer_model == INHOUSE_LLM_MODEL
    assert args.guard_classifier == "openai"
    assert args.classifier_model == INHOUSE_LLM_MODEL
    assert args.entailment_verifier == "openai"
    assert args.entailment_model == INHOUSE_LLM_MODEL
    assert args.evidence_min_score == INHOUSE_EVIDENCE_MIN_SCORE
    assert args.policy.name == "guardrail_policy_bge_m3.toml"


def test_inhouse_profile_rejects_missing_custom_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_URL", raising=False)

    with pytest.raises(InHouseEndpointError, match="Fraunhofer"):
        apply_model_profile(_runtime_args())


def test_inhouse_profile_rejects_official_openai_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    with pytest.raises(InHouseEndpointError, match="api.openai.com"):
        apply_model_profile(_runtime_args())


def test_local_profile_keeps_existing_cli_defaults() -> None:
    args = _runtime_args()
    args.profile = "local"

    apply_model_profile(args)

    assert args.retriever == "lexical"
    assert args.embedding_provider == "hashing"
    assert args.generator == "extractive"
    assert args.guard_classifier == "none"


def test_profile_summary_never_contains_url_or_key(monkeypatch) -> None:
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://learning-services4.fokus.fraunhofer.de/litellm/v1",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "not-for-output")

    summary = model_profile_summary("inhouse")

    assert summary["endpoint_host"] == "learning-services4.fokus.fraunhofer.de"
    assert summary["retrieval_evidence_threshold"] == INHOUSE_EVIDENCE_MIN_SCORE
    serialized = str(summary)
    assert "not-for-output" not in serialized
    assert "https://" not in serialized
