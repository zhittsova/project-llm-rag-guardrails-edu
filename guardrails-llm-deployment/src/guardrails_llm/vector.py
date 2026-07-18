from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from .corpus import Chunk, JsonMetadata, load_documents
from .embeddings import HASHING_EMBEDDING_MODEL, TextEmbedder, create_embedder, resolve_embedding_model
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
    # build-index pipeline:
    # 1. загрузить JSONL corpus;
    # 2. разрезать documents на chunks через LangChain splitter;
    # 3. превратить каждый chunk в embedding;
    # 4. сохранить chunks + metadata в persistent Chroma collection.
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
            "collection": COLLECTION_NAME,
            "embedding_provider": embedding_provider,
            "embedding_model": resolved_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
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
    ) -> None:
        resolved_model = resolve_embedding_model(embedding_provider, embedding_model)
        _assert_manifest_matches(index_dir, embedding_provider, resolved_model)
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

        # Query проходит через ту же embedding-функцию, что и chunks при
        # build-index. Chroma возвращает ближайшие chunks по cosine distance.
        query_options: dict[str, object] = {
            "query_embeddings": [self._embedder.embed(query)],
            "n_results": min(top_k, count),
            "include": ["documents", "metadatas", "distances"],
        }
        where = _chroma_where(course_id, allowed_visibility)
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
            # Фильтры оставлены на уровне retriever interface, чтобы lexical,
            # LangChain и vector backends вели себя одинаково для pipeline.
            if course_id and chunk.course_id != course_id:
                continue
            if allowed_visibility and chunk.visibility not in allowed_visibility:
                continue
            score = 1.0 - float(distance)
            if score >= self._min_score:
                matches.append((chunk, score))
            if len(matches) == top_k:
                break
        return matches


def _chroma_where(
    course_id: str | None,
    allowed_visibility: set[str] | None,
) -> dict[str, object] | None:
    filters: list[dict[str, object]] = []
    if course_id:
        filters.append({"course_id": {"$eq": course_id}})
    if allowed_visibility:
        filters.append({"visibility": {"$in": sorted(allowed_visibility)}})
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


def _assert_manifest_matches(index_dir: Path, provider: str, model: str) -> None:
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
