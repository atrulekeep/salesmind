"""路由：/health、/api/retrieve、/api/chat/stream（SSE）。"""
import json
from typing import Literal

import httpx
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sse_starlette import EventSourceResponse, ServerSentEvent

from app.schemas import RetrievedChunk

router = APIRouter()


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: Literal["vector", "hybrid"] = "hybrid"
    use_rerank: bool = True
    rewrite_mode: Literal["none", "multi", "hyde", "both"] = "none"
    topk: int = Field(default=5, ge=1, le=20)


class ChatRequest(RetrieveRequest):
    """与 RetrieveRequest 同形：问题 + 检索开关。"""


def _item_dict(it: RetrievedChunk) -> dict:
    """RetrievedChunk → 响应字典（含调试分数与正文）。"""
    return {
        "chunk_id": it.chunk.chunk_id,
        "title": it.chunk.title,
        "source": it.chunk.source,
        "section_path": it.chunk.section_path,
        "valid_from": it.chunk.valid_from,
        "doc_version": it.chunk.doc_version,
        "chunk_type": it.chunk.chunk_type,
        "content": it.chunk.content,
        "final_rank": it.final_rank,
        "bm25_rank": it.bm25_rank,
        "vector_rank": it.vector_rank,
        "rrf_score": it.rrf_score,
        "rerank_score": it.rerank_score,
    }


async def _ollama_alive(base_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
            return (await client.get(base_url.rstrip("/") + "/api/tags")).status_code == 200
    except httpx.HTTPError:
        return False


@router.get("/health")
async def health(request: Request):
    state = request.app.state
    alive = await _ollama_alive(state.settings.ollama_base_url)
    n_chunks = len(state.bm.chunks)
    return {
        "status": "ok" if alive and n_chunks else "degraded",
        "ollama": "ok" if alive else "unreachable",
        "index": "ready" if n_chunks else "missing",
        "chunks": n_chunks,
        "reranker": state.reranker.status,
    }


@router.post("/api/retrieve")
async def retrieve(req: RetrieveRequest, request: Request):
    result = await request.app.state.retriever.retrieve(
        req.question, mode=req.mode, use_rerank=req.use_rerank,
        rewrite_mode=req.rewrite_mode, topk=req.topk,
    )
    return {
        "items": [_item_dict(it) for it in result.items],
        "rerank_available": result.rerank_available,
        "trace": result.trace.model_dump(),
    }


@router.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """SSE：status → retrieved → answer_delta* → [status(citation_retry) → answer_delta*] → done。

    事件 id 单调递增（为 Last-Event-ID 续传预留，当前不实现续传）。
    """
    state = request.app.state

    async def event_gen():
        seq = 0

        def sse(event: str, data: dict) -> ServerSentEvent:
            nonlocal seq
            seq += 1
            return ServerSentEvent(
                id=str(seq), event=event,
                data=json.dumps(data, ensure_ascii=False),
            )

        yield sse("status", {"stage": "retrieving"})
        result = await state.retriever.retrieve(
            req.question, mode=req.mode, use_rerank=req.use_rerank,
            rewrite_mode=req.rewrite_mode, topk=req.topk,
        )
        yield sse("retrieved", {
            "items": [_item_dict(it) for it in result.items],
            "rerank_available": result.rerank_available,
            "rewrite_fell_back": result.trace.rewrite_fell_back,
        })
        async for ev in state.generator.generate(req.question, result.items):
            if ev["type"] == "answer_delta":
                yield sse("answer_delta", {"text": ev["text"]})
            elif ev["type"] == "status":
                yield sse("status", {"stage": ev["stage"], "detail": ev["detail"]})
            else:  # done
                trace = result.trace
                yield sse("done", {
                    "answer": ev["answer"],
                    "citations": ev["citations"],
                    "citation_ok": ev["citation_ok"],
                    "refused": ev["refused"],
                    "usage": ev["usage"],
                    "latency_ms": ev["latency_ms"],
                    "trace": {
                        "timings_ms": trace.timings_ms,
                        "fused": trace.fused,
                        "rerank_available": result.rerank_available,
                        "rewrite_fell_back": trace.rewrite_fell_back,
                    },
                })

    return EventSourceResponse(event_gen())
