"""
RAG System (ICE Flow)
=====================

This script implements a simple RAG pipeline with three stages:
1. Ingest (I): Load PDFs, split into chunks, embed, and upsert into Qdrant.
2. Context (C): Search for relevant chunks given a query.
3. Explain (E): Use an LLM to generate an answer based on retrieved context.

Components:
- FastAPI app to serve Inngest functions.
- Inngest functions for ingestion and querying.
- Qdrant vector database for storage and retrieval.
"""

# -------------------- Imports --------------------
import os
import uuid
import datetime
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai

# Local modules
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult


# -------------------- Environment --------------------
load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


# -------------------- Ingest PDFs --------------------
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(limit=2, period=datetime.timedelta(minutes=1)),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
    ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    """
    Ingest a PDF:
    1. Load and chunk text.
    2. Embed text chunks.
    3. Upsert into Qdrant.
    """

    def _load() -> RAGChunkAndSrc:
        pdf = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf)
        return RAGChunkAndSrc(chunks=load_and_chunk_pdf(pdf), source_id=source_id)

    def _upsert(data: RAGChunkAndSrc) -> RAGUpsertResult:
        vecs = embed_texts(data.chunks)
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{data.source_id}:{i}"))
            for i in range(len(data.chunks))
        ]
        payloads = [{"source": data.source_id, "text": chunk} for chunk in data.chunks]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(data.chunks))

    data = await ctx.step.run("load-and-chunk", _load, output_type=RAGChunkAndSrc)
    result = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(data),
        output_type=RAGUpsertResult,
    )
    return result.model_dump()


# -------------------- Query PDFs (Context + Explain) --------------------
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    """
    Query the RAG system:
    1. Embed the question and search Qdrant.
    2. Construct a prompt with retrieved context.
    3. Use an LLM to generate an answer.
    """

    def _search(q: str, top_k: int) -> RAGSearchResult:
        vec = embed_texts([q])[0]
        found = QdrantStorage().search(vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult,
    )

    context_text = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        f"Answer concisely using only this context."
    )

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPEN_API_KEY"),
        model="gpt-4o-mini",
    )
    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "Answer using only the provided context."},
                {"role": "user", "content": user_content},
            ],
        },
    )

    return {
        "answer": res["choices"][0]["message"]["content"].strip(),
        "sources": found.sources,
        "num_contexts": len(found.contexts),
    }


# -------------------- FastAPI App --------------------
app = FastAPI()
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
