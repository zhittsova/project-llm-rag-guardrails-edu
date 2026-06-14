from __future__ import annotations

from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .corpus import Chunk, Document
from .retrieval import LexicalRetriever


def langchain_chunk_documents(documents: list[Document], *, chunk_size: int = 650, chunk_overlap: int = 80) -> list[Chunk]:
    # LangChain здесь не используется как LLM. Он нужен как готовая обвязка для
    # Document objects и RecursiveCharacterTextSplitter, чтобы аккуратно резать
    # длинные course documents на chunks с overlap.
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    langchain_documents = [
        LangChainDocument(
            page_content=document.text,
            metadata={
                "doc_id": document.doc_id,
                "course_id": document.course_id,
                "title": document.title,
                "visibility": document.visibility,
                "source_type": document.source_type,
                **document.metadata,
            },
        )
        for document in documents
    ]
    split_documents = splitter.split_documents(langchain_documents)
    chunks: list[Chunk] = []
    per_doc_counts: dict[str, int] = {}
    for split_document in split_documents:
        metadata = split_document.metadata
        doc_id = str(metadata["doc_id"])
        index = per_doc_counts.get(doc_id, 0)
        per_doc_counts[doc_id] = index + 1
        # После LangChain split возвращаемся к нашему собственному Chunk type.
        # Так pipeline не зависит напрямую от LangChain и может работать с
        # lexical или vector retriever через один интерфейс.
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}:lc:{index}",
                doc_id=doc_id,
                course_id=str(metadata["course_id"]),
                title=str(metadata["title"]),
                visibility=str(metadata["visibility"]),
                source_type=str(metadata["source_type"]),
                text=split_document.page_content,
                metadata={
                    key: value
                    for key, value in metadata.items()
                    if key not in {"doc_id", "course_id", "title", "visibility", "source_type"}
                },
            )
        )
    return chunks


class LangChainLexicalRetriever(LexicalRetriever):
    """LangChain document pipeline with the project's deterministic scorer.

    This keeps the Workshop 2 demo reproducible while using LangChain's document
    abstraction and text splitter. A vector store or LLM can be swapped in later
    without changing the assistant/evaluation boundary.
    """

    @classmethod
    def from_documents(cls, documents: list[Document]) -> "LangChainLexicalRetriever":
        return cls(langchain_chunk_documents(documents))
