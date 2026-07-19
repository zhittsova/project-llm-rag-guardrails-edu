from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import chromadb

from .corpus import Chunk, JsonMetadata, load_documents
from .embeddings import HASHING_EMBEDDING_MODEL, TextEmbedder, create_embedder, resolve_embedding_model
from .grounding import POLICY_SOURCE_TYPES
from .langchain_rag import langchain_chunk_documents
from .retrieval import tokenize


COLLECTION_NAME = "course_chunks"
MANIFEST_NAME = "course_chunks_manifest.json"
DEFAULT_VECTOR_MIN_SCORE = 0.05
REQUIRED_METADATA = {"chunk_id", "doc_id", "course_id", "title", "visibility", "source_type"}


@dataclass(frozen=True)
class VectorIndexStats:
    corpus: Path
    index_dir: Path
    collection: str
    documents: int
    chunks: int
    embedding_provider: str
    embedding_model: str


class VectorIndexError(RuntimeError):
    pass


class VectorIndexNotFoundError(VectorIndexError):
    pass


class VectorIndexConfigurationError(VectorIndexError):
    pass


def default_index_path() -> Path:
    return Path(__file__).resolve().parents[2] / "indexes" / "chroma"


def build_vector_index(
    corpus_path: Path,
    index_dir: Path,
    *,
    chunk_size: int = 650,
    chunk_overlap: int = 80,
    embedding_provider: str = "hashing",
    embedding_model: str | None = None,
    allow_remote_models: bool = False,
    env_file: Path | None = None,
    embedding_cache_path: Path | None = None,
    embedder: TextEmbedder | None = None,
) -> VectorIndexStats:
    # Indexing loads the JSONL corpus, splits documents with LangChain, embeds
    # each chunk, and stores the chunks and metadata in persistent Chroma.
    resolved_model = resolve_embedding_model(embedding_provider, embedding_model)
    embedder = embedder or create_embedder(
        embedding_provider,
        model=resolved_model,
        allow_remote_models=allow_remote_models,
        env_file=env_file,
        cache_path=embedding_cache_path,
    )
    documents = load_documents(corpus_path)
    chunks = langchain_chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_many(chunk_texts) if chunk_texts else []

    client = _persistent_client(index_dir)
    _delete_collection_if_present(client, COLLECTION_NAME)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if chunks:
        collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=[_metadata_for_chroma(chunk) for chunk in chunks],
        )
    _write_manifest(
        index_dir,
        {
            "schema_version": 2,
            "collection": COLLECTION_NAME,
            "embedding_provider": embedding_provider,
            "embedding_model": resolved_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "corpus_sha256": _file_sha256(corpus_path),
            "chunk_count": len(chunks),
            "chunks_sha256": _chunks_sha256(chunks),
        },
    )

    return VectorIndexStats(
        corpus=corpus_path,
        index_dir=index_dir,
        collection=COLLECTION_NAME,
        documents=len(documents),
        chunks=len(chunks),
        embedding_provider=embedding_provider,
        embedding_model=resolved_model,
    )


class VectorRetriever:
    def __init__(
        self,
        index_dir: Path,
        *,
        min_score: float = DEFAULT_VECTOR_MIN_SCORE,
        embedding_provider: str = "hashing",
        embedding_model: str | None = None,
        allow_remote_models: bool = False,
        env_file: Path | None = None,
        embedding_cache_path: Path | None = None,
        embedder: TextEmbedder | None = None,
        corpus_path: Path | None = None,
        policy_context_top_k: int = 0,
        policy_context_min_score: float = 0.0,
    ) -> None:
        if policy_context_top_k < 0:
            raise ValueError("policy_context_top_k must be non-negative")
        if not math.isfinite(policy_context_min_score):
            raise ValueError("policy_context_min_score must be finite")
        resolved_model = resolve_embedding_model(embedding_provider, embedding_model)
        _assert_manifest_matches(
            index_dir,
            embedding_provider,
            resolved_model,
            corpus_path=corpus_path,
        )
        try:
            self._collection = _persistent_client(index_dir).get_collection(COLLECTION_NAME)
        except chromadb.errors.NotFoundError as exc:
            raise VectorIndexNotFoundError(
                f"Vector index at {index_dir} ({index_dir.resolve()}) does not contain collection "
                f"{COLLECTION_NAME!r}. Run build-index first, or use "
                "`./scripts/run_workshop2_demo.sh` for the full demo flow."
            ) from exc
        self._embedder = embedder or create_embedder(
            embedding_provider,
            model=resolved_model,
            allow_remote_models=allow_remote_models,
            env_file=env_file,
            cache_path=embedding_cache_path,
        )
        self._min_score = min_score
        self._policy_context_top_k = policy_context_top_k
        self._policy_context_min_score = policy_context_min_score

    @property
    def indexed_chunks(self) -> int:
        return self._collection.count()

    def search(
        self,
        query: str,
        *,
        course_id: str | None = None,
        allowed_visibility: set[str] | None = None,
        top_k: int = 3,
    ) -> list[tuple[Chunk, float]]:
        if not tokenize(query):
            return []

        count = self.indexed_chunks
        if count == 0:
            return []

        # Embed the query with the same function used at indexing time. Chroma
        # returns the chunks with the smallest cosine distance.
        query_embedding = self._embedder.embed(query)
        matches = self._query(
            query_embedding,
            where=_chroma_where(course_id, allowed_visibility),
            top_k=top_k,
            min_score=self._min_score,
        )
        if self._policy_context_top_k and allowed_visibility is not None:
            policy_matches = self._query(
                query_embedding,
                where=_chroma_where(
                    course_id,
                    allowed_visibility,
                    allowed_source_types=POLICY_SOURCE_TYPES,
                ),
                top_k=self._policy_context_top_k,
                min_score=max(self._min_score, self._policy_context_min_score),
            )
            seen = {chunk.chunk_id for chunk, _score in matches}
            matches.extend(
                (chunk, score)
                for chunk, score in policy_matches
                if chunk.chunk_id not in seen
            )
        return matches

    def _query(
        self,
        query_embedding: list[float],
        *,
        where: dict[str, object] | None,
        top_k: int,
        min_score: float,
    ) -> list[tuple[Chunk, float]]:
        query_options: dict[str, object] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.indexed_chunks),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_options["where"] = where
        results = self._collection.query(**query_options)
        matches: list[tuple[Chunk, float]] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for text, metadata, distance in zip(documents, metadatas, distances, strict=True):
            if metadata is None:
                continue
            chunk = _chunk_from_chroma(text or "", metadata)
            score = 1.0 - float(distance)
            if score >= min_score:
                matches.append((chunk, score))
            if len(matches) == top_k:
                break
        return matches


def _chroma_where(
    course_id: str | None,
    allowed_visibility: set[str] | None,
    allowed_source_types: set[str] | frozenset[str] | None = None,
) -> dict[str, object] | None:
    filters: list[dict[str, object]] = []
    if course_id:
        filters.append({"course_id": {"$eq": course_id}})
    if allowed_visibility:
        filters.append({"visibility": {"$in": sorted(allowed_visibility)}})
    if allowed_source_types:
        filters.append({"source_type": {"$in": sorted(allowed_source_types)}})
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def _persistent_client(index_dir: Path) -> chromadb.PersistentClient:
    index_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(index_dir))


def _manifest_path(index_dir: Path) -> Path:
    return index_dir / MANIFEST_NAME


def _write_manifest(index_dir: Path, manifest: dict[str, object]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    _manifest_path(index_dir).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _read_manifest(index_dir: Path) -> dict[str, object] | None:
    path = _manifest_path(index_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise VectorIndexConfigurationError(f"Vector index manifest at {path} must be a JSON object.")
    return data


def _assert_manifest_matches(
    index_dir: Path,
    provider: str,
    model: str,
    *,
    corpus_path: Path | None = None,
) -> None:
    manifest = _read_manifest(index_dir)
    if manifest is None:
        if provider == "hashing" and model == HASHING_EMBEDDING_MODEL:
            return
        raise VectorIndexConfigurationError(
            f"Vector index at {index_dir} has no embedding manifest. Rebuild it with "
            f"--embedding-provider {provider} before querying with {model}."
        )
    indexed_provider = manifest.get("embedding_provider")
    indexed_model = manifest.get("embedding_model")
    if indexed_provider != provider or indexed_model != model:
        raise VectorIndexConfigurationError(
            f"Vector index at {index_dir} was built with {indexed_provider}/{indexed_model}, "
            f"but the query requested {provider}/{model}. Rebuild the index or pass matching "
            "embedding options."
        )
    if corpus_path is not None:
        indexed_corpus = manifest.get("corpus_sha256")
        requested_corpus = _file_sha256(corpus_path)
        if indexed_corpus != requested_corpus:
            raise VectorIndexConfigurationError(
                f"Vector index at {index_dir} was built from a different corpus. "
                "Rebuild the index before querying this corpus."
            )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunks_sha256(chunks: list[Chunk]) -> str:
    digest = sha256()
    for chunk in chunks:
        payload = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "course_id": chunk.course_id,
            "title": chunk.title,
            "visibility": chunk.visibility,
            "source_type": chunk.source_type,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
        digest.update(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _delete_collection_if_present(client: chromadb.PersistentClient, name: str) -> None:
    for collection in client.list_collections():
        collection_name = collection.name if hasattr(collection, "name") else str(collection)
        if collection_name == name:
            client.delete_collection(name)
            return


def _metadata_for_chroma(chunk: Chunk) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "course_id": chunk.course_id,
        "title": chunk.title,
        "visibility": chunk.visibility,
        "source_type": chunk.source_type,
    }
    for key, value in chunk.metadata.items():
        encoded = _encode_metadata(value)
        if encoded is not None:
            metadata[key] = encoded
    return metadata


def _chunk_from_chroma(text: str, metadata: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=str(metadata["chunk_id"]),
        doc_id=str(metadata["doc_id"]),
        course_id=str(metadata["course_id"]),
        title=str(metadata["title"]),
        visibility=str(metadata["visibility"]),
        source_type=str(metadata["source_type"]),
        text=text,
        metadata={
            key: _decode_metadata(key, value)
            for key, value in metadata.items()
            if key not in REQUIRED_METADATA
        },
    )


def _encode_metadata(value: JsonMetadata) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, list):
        return json.dumps(value)
    return value


def _decode_metadata(key: str, value: Any) -> JsonMetadata:
    if key == "allowed_audience" and isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(decoded, list) and all(isinstance(item, str) for item in decoded):
            return decoded
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)
