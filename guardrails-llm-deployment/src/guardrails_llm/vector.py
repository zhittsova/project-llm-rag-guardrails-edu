from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any

import chromadb

from .corpus import Chunk, JsonMetadata, load_documents
from .langchain_rag import langchain_chunk_documents
from .retrieval import tokenize


COLLECTION_NAME = "course_chunks"
REQUIRED_METADATA = {"chunk_id", "doc_id", "course_id", "title", "visibility", "source_type"}


@dataclass(frozen=True)
class VectorIndexStats:
    corpus: Path
    index_dir: Path
    collection: str
    documents: int
    chunks: int


class VectorIndexNotFoundError(RuntimeError):
    pass


class HashingEmbedder:
    # Локальная deterministic embedding-функция для demo: без API keys,
    # downloads и случайности. Это не production semantic embedding model, но
    # она создает числовой vector, с которым Chroma может строить index.
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def default_index_path() -> Path:
    return Path(__file__).resolve().parents[2] / "indexes" / "chroma"


def build_vector_index(
    corpus_path: Path,
    index_dir: Path,
    *,
    chunk_size: int = 650,
    chunk_overlap: int = 80,
) -> VectorIndexStats:
    # build-index pipeline:
    # 1. загрузить JSONL corpus;
    # 2. разрезать documents на chunks через LangChain splitter;
    # 3. превратить каждый chunk в embedding;
    # 4. сохранить chunks + metadata в persistent Chroma collection.
    documents = load_documents(corpus_path)
    chunks = langchain_chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    client = _persistent_client(index_dir)
    _delete_collection_if_present(client, COLLECTION_NAME)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if chunks:
        embedder = HashingEmbedder()
        collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embedder.embed_many([chunk.text for chunk in chunks]),
            metadatas=[_metadata_for_chroma(chunk) for chunk in chunks],
        )

    return VectorIndexStats(
        corpus=corpus_path,
        index_dir=index_dir,
        collection=COLLECTION_NAME,
        documents=len(documents),
        chunks=len(chunks),
    )


class VectorRetriever:
    def __init__(self, index_dir: Path, *, min_score: float = 0.05) -> None:
        try:
            self._collection = _persistent_client(index_dir).get_collection(COLLECTION_NAME)
        except chromadb.errors.NotFoundError as exc:
            raise VectorIndexNotFoundError(
                f"Vector index at {index_dir} does not contain collection "
                f"{COLLECTION_NAME!r}. Run build-index first, or use "
                "`./scripts/run_workshop2_demo.sh` for the full demo flow."
            ) from exc
        self._embedder = HashingEmbedder()
        self._min_score = min_score

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

        count = self._collection.count()
        if count == 0:
            return []

        # Query проходит через ту же embedding-функцию, что и chunks при
        # build-index. Chroma возвращает ближайшие chunks по cosine distance.
        results = self._collection.query(
            query_embeddings=[self._embedder.embed(query)],
            n_results=count,
            include=["documents", "metadatas", "distances"],
        )
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


def _persistent_client(index_dir: Path) -> chromadb.PersistentClient:
    index_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(index_dir))


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
