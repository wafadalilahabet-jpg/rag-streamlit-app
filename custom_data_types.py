"""
RAG Schemas
===========

This module defines Pydantic models used for:
1. Representing document chunks and sources.
2. Capturing results of vector upsert operations.
3. Structuring search results from the vector database.
4. Returning query answers with sources and context counts.
"""

# -------------------- Imports --------------------
from pydantic import BaseModel


# -------------------- Data Models --------------------
class RAGChunkAndSrc(BaseModel):
    """
    Represents a set of text chunks extracted from a document,
    along with the document's source identifier.
    """
    chunks: list[str]
    source_id: str | None = None


class RAGUpsertResult(BaseModel):
    """
    Result of an upsert operation into the vector database.
    """
    ingested: int


class RAGSearchResult(BaseModel):
    """
    Result of a similarity search in the vector database.
    """
    contexts: list[str]
    sources: list[str]


class RAGQueryResult(BaseModel):
    """
    Final result returned by the RAG pipeline after answering a query.
    """
    answer: str
    sources: list[str]
    num_contexts: int
