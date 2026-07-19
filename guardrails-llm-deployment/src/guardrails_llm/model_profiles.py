from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from urllib.parse import urlparse

from .guardrail_policy import default_policy_path
from .model_config import (
    OpenAIModelConfig,
    load_local_env,
    resolve_openai_base_url,
    resolve_openai_base_url_source,
)


LOCAL_PROFILE = "local"
INHOUSE_PROFILE = "inhouse"
MODEL_PROFILES = (LOCAL_PROFILE, INHOUSE_PROFILE)
INHOUSE_ENDPOINT_HOST = "learning-services4.fokus.fraunhofer.de"
INHOUSE_EMBEDDING_MODEL = "BAAI/bge-m3"
INHOUSE_LLM_MODEL = "Qwen/Qwen3.6-35B-A3B"
INHOUSE_EVIDENCE_MIN_SCORE = 0.5618841052055359
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INHOUSE_EMBEDDING_CACHE = PROJECT_ROOT / "indexes" / "cache" / "bge-m3.jsonl"
INHOUSE_INDEX_DIR = PROJECT_ROOT / "indexes" / "python-course-bge-m3"
INHOUSE_CORPUS_PATH = PROJECT_ROOT / "data" / "python_course_docs.jsonl"
INHOUSE_COURSE_ID = "python-intro"


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
    _set_if_present(args, "answer_model", INHOUSE_LLM_MODEL)
    _set_if_present(args, "guard_classifier", "openai")
    _set_if_present(args, "classifier_model", INHOUSE_LLM_MODEL)
    _set_if_present(args, "entailment_verifier", "openai")
    _set_if_present(args, "entailment_model", INHOUSE_LLM_MODEL)
    if (
        hasattr(args, "evidence_min_score")
        and args.evidence_min_score is None
    ):
        args.evidence_min_score = INHOUSE_EVIDENCE_MIN_SCORE
    _set_if_present(args, "judge_model", INHOUSE_LLM_MODEL)
    if hasattr(args, "policy") and args.policy is None:
        args.policy = default_policy_path().with_name("guardrail_policy_bge_m3.toml")


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
        "answer_model": INHOUSE_LLM_MODEL,
        "classifier_model": INHOUSE_LLM_MODEL,
        "entailment_model": INHOUSE_LLM_MODEL,
        "api_key_present": _api_key_present(config),
        "remote_calls_require_explicit_allowance": True,
        "retrieval_evidence_threshold": INHOUSE_EVIDENCE_MIN_SCORE,
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
