from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from urllib.parse import urlparse

from .guardrail_runtime import (
    PROJECT_ROOT,
    default_inhouse_runtime_path,
    load_guardrail_runtime_config,
    runtime_config_sha256,
)
from .model_config import (
    OpenAIModelConfig,
    load_local_env,
    resolve_openai_base_url,
    resolve_openai_base_url_source,
)


LOCAL_PROFILE = "local"
INHOUSE_PROFILE = "inhouse"
MODEL_PROFILES = (LOCAL_PROFILE, INHOUSE_PROFILE)
INHOUSE_RUNTIME_CONFIG_PATH = default_inhouse_runtime_path()
INHOUSE_RUNTIME_CONFIG = load_guardrail_runtime_config(
    INHOUSE_RUNTIME_CONFIG_PATH
)
INHOUSE_ENDPOINT_HOST = INHOUSE_RUNTIME_CONFIG.endpoint_host
INHOUSE_EMBEDDING_MODEL = INHOUSE_RUNTIME_CONFIG.models.embedding
INHOUSE_LLM_MODEL = INHOUSE_RUNTIME_CONFIG.models.answer
INHOUSE_EVIDENCE_MIN_SCORE = INHOUSE_RUNTIME_CONFIG.retrieval.evidence_min_score
INHOUSE_RETRIEVAL_TOP_K = INHOUSE_RUNTIME_CONFIG.retrieval.top_k
INHOUSE_POLICY_CONTEXT_TOP_K = INHOUSE_RUNTIME_CONFIG.retrieval.policy_context_top_k
INHOUSE_POLICY_CONTEXT_MIN_SCORE = (
    INHOUSE_RUNTIME_CONFIG.retrieval.policy_context_min_score
)
INHOUSE_CLASSIFIER_MIN_CONFIDENCE = (
    INHOUSE_RUNTIME_CONFIG.classifier.min_confidence
)
INHOUSE_ENTAILMENT_MIN_CONFIDENCE = (
    INHOUSE_RUNTIME_CONFIG.retrieval.entailment_min_confidence
)
INHOUSE_RESOLVED_PATHS = INHOUSE_RUNTIME_CONFIG.paths.resolve(PROJECT_ROOT)
INHOUSE_EMBEDDING_CACHE = INHOUSE_RESOLVED_PATHS.embedding_cache
INHOUSE_INDEX_DIR = INHOUSE_RESOLVED_PATHS.index
INHOUSE_CORPUS_PATH = INHOUSE_RESOLVED_PATHS.corpus
INHOUSE_POLICY_PATH = INHOUSE_RESOLVED_PATHS.policy
INHOUSE_COURSE_ID = INHOUSE_RUNTIME_CONFIG.course_id


class InHouseEndpointError(RuntimeError):
    pass


def apply_model_profile(args: Namespace) -> None:
    profile = getattr(args, "profile", LOCAL_PROFILE)
    if profile == LOCAL_PROFILE:
        return
    if profile != INHOUSE_PROFILE:
        raise ValueError(f"unknown model profile: {profile}")

    _require_inhouse_endpoint(getattr(args, "env_file", None))
    _set_if_present(args, "retriever", "vector")
    if getattr(args, "command", None) in {
        "query",
        "evaluate",
        "compare-guardrails",
        "build-index",
        "visualize",
    }:
        _set_if_present(args, "index_dir", INHOUSE_INDEX_DIR)
        _set_if_present(args, "course_id", INHOUSE_COURSE_ID)
        if hasattr(args, "command_corpus"):
            args.command_corpus = INHOUSE_CORPUS_PATH
    _set_if_present(args, "embedding_provider", "openai")
    _set_if_present(args, "embedding_model", INHOUSE_EMBEDDING_MODEL)
    if hasattr(args, "embedding_cache") and args.embedding_cache is None:
        args.embedding_cache = INHOUSE_EMBEDDING_CACHE
    _set_if_present(args, "guard_embedding_provider", "openai")
    _set_if_present(args, "guard_embedding_model", INHOUSE_EMBEDDING_MODEL)
    _set_if_present(args, "generator", "openai")
    _set_if_present(args, "answer_model", INHOUSE_RUNTIME_CONFIG.models.answer)
    _set_if_present(args, "guard_classifier", "openai")
    _set_if_present(
        args,
        "classifier_model",
        INHOUSE_RUNTIME_CONFIG.models.classifier,
    )
    _set_if_present(
        args,
        "classifier_strategy",
        INHOUSE_RUNTIME_CONFIG.classifier.strategy,
    )
    _set_if_present(
        args,
        "classifier_min_confidence",
        INHOUSE_CLASSIFIER_MIN_CONFIDENCE,
    )
    _set_if_present(args, "entailment_verifier", "openai")
    _set_if_present(
        args,
        "entailment_model",
        INHOUSE_RUNTIME_CONFIG.models.entailment,
    )
    _set_if_present(
        args,
        "entailment_min_confidence",
        INHOUSE_ENTAILMENT_MIN_CONFIDENCE,
    )
    _set_if_present(args, "retrieval_top_k", INHOUSE_RETRIEVAL_TOP_K)
    _set_if_present(args, "policy_context_top_k", INHOUSE_POLICY_CONTEXT_TOP_K)
    _set_if_present(args, "policy_context_min_score", INHOUSE_POLICY_CONTEXT_MIN_SCORE)
    if (
        hasattr(args, "evidence_min_score")
        and args.evidence_min_score is None
    ):
        args.evidence_min_score = INHOUSE_EVIDENCE_MIN_SCORE
    _set_if_present(args, "judge_model", INHOUSE_RUNTIME_CONFIG.models.judge)
    if hasattr(args, "policy") and args.policy is None:
        args.policy = INHOUSE_POLICY_PATH


def model_profile_summary(
    profile: str,
    env_file: Path | None = None,
) -> dict[str, object]:
    if profile == LOCAL_PROFILE:
        return {
            "profile": LOCAL_PROFILE,
            "remote_models": False,
            "embedding_provider": "hashing",
            "embedding_model": "hashing-blake2b-384",
            "generator": "extractive",
            "classifier": "none",
            "entailment_verifier": "none",
        }
    if profile != INHOUSE_PROFILE:
        raise ValueError(f"unknown model profile: {profile}")

    config, host = _require_inhouse_endpoint(env_file)
    return {
        "profile": INHOUSE_PROFILE,
        "remote_models": True,
        "endpoint_host": host,
        "endpoint_env": resolve_openai_base_url_source(config),
        "embedding_provider": "openai_compatible",
        "embedding_model": INHOUSE_EMBEDDING_MODEL,
        "course_id": INHOUSE_COURSE_ID,
        "index_directory": INHOUSE_INDEX_DIR.name,
        "answer_model": INHOUSE_RUNTIME_CONFIG.models.answer,
        "classifier_model": INHOUSE_RUNTIME_CONFIG.models.classifier,
        "entailment_model": INHOUSE_RUNTIME_CONFIG.models.entailment,
        "judge_model": INHOUSE_RUNTIME_CONFIG.models.judge,
        "api_key_present": _api_key_present(config),
        "remote_calls_require_explicit_allowance": True,
        "retrieval_top_k": INHOUSE_RETRIEVAL_TOP_K,
        "retrieval_evidence_threshold": INHOUSE_EVIDENCE_MIN_SCORE,
        "policy_context_top_k": INHOUSE_POLICY_CONTEXT_TOP_K,
        "policy_context_min_score": INHOUSE_POLICY_CONTEXT_MIN_SCORE,
        "classifier_min_confidence": INHOUSE_CLASSIFIER_MIN_CONFIDENCE,
        "entailment_min_confidence": INHOUSE_ENTAILMENT_MIN_CONFIDENCE,
        "runtime_config_schema_version": INHOUSE_RUNTIME_CONFIG.schema_version,
        "runtime_config_sha256": runtime_config_sha256(
            INHOUSE_RUNTIME_CONFIG_PATH
        ),
    }


def ensure_inhouse_endpoint(env_file: Path | None = None) -> str:
    _config, host = _require_inhouse_endpoint(env_file)
    return host


def _require_inhouse_endpoint(
    env_file: Path | None,
) -> tuple[OpenAIModelConfig, str]:
    config = OpenAIModelConfig(env_file=env_file)
    load_local_env(config.env_file)
    url = resolve_openai_base_url(config)
    host = urlparse(url).hostname if url else None
    if host != INHOUSE_ENDPOINT_HOST:
        actual = host or "not configured"
        raise InHouseEndpointError(
            "The inhouse profile requires the Fraunhofer endpoint host "
            f"{INHOUSE_ENDPOINT_HOST}; resolved host: {actual}."
        )
    return config, host


def _api_key_present(config: OpenAIModelConfig) -> bool:
    import os

    return bool(os.getenv(config.api_key_env))


def _set_if_present(args: Namespace, name: str, value: object) -> None:
    if hasattr(args, name):
        setattr(args, name, value)
