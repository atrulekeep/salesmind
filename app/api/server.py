"""FastAPI 入口：lifespan 装配组件 + create_app 工厂。

组件工厂可注入（测试用 fake 替换真实索引/LLM，见 tests/test_api.py）。
同一套检索核心供 CLI / API / MCP / 评测复用，业务逻辑零拷贝。
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.routes import router
from app.cli import _build_components
from app.generation.answer import AnswerGenerator
from app.indexing.pipeline import IndexPipeline
from app.retrieval.retriever import Retriever
from app.retrieval.rewrite import Rewriter


def create_app(components=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s, embedder, vs, bm, mf, llm, reranker = components or _build_components()
        app.state.settings = s
        app.state.bm = bm
        app.state.llm = llm
        app.state.reranker = reranker
        app.state.manifest = mf
        app.state.pipeline = IndexPipeline(s, embedder, vs, bm, mf)
        app.state.index_lock = asyncio.Lock()
        app.state.retriever = Retriever(
            embedder, vs, bm, reranker=reranker, rewriter=Rewriter(llm), settings=s,
        )
        app.state.generator = AnswerGenerator(llm)
        yield

    app = FastAPI(title="SalesMind", version="0.1.0", lifespan=lifespan)
    app.include_router(router)
    app.include_router(documents_router)
    return app
