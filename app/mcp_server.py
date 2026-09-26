"""SalesMind MCP Server（stdio）：只读知识检索工具。

复用与 CLI / FastAPI 相同的检索核心（_build_components → Retriever），
业务逻辑零拷贝。不暴露写工具。
"""
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from app.cli import _build_components
from app.retrieval.retriever import Retriever
from app.retrieval.rewrite import Rewriter

mcp = FastMCP("salesmind")

# 模块级初始化：加载索引 + 构建检索器（进程生命周期内复用）
_s, _embedder, _vs, _bm, _mf, _llm, _reranker = _build_components()
_retriever = Retriever(
    _embedder, _vs, _bm,
    reranker=_reranker, rewriter=Rewriter(_llm), settings=_s,
)


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="检索问题")
    top_k: int = Field(default=5, ge=1, le=20, description="返回条数")
    acl: str = Field(default="public", description="密级：public | internal")


@mcp.tool()
async def knowledge_search(query: str, top_k: int = 5, acl: str = "public") -> list[dict]:
    """检索智策云CRM销售知识库，返回要点与引用来源（只读）。

    每个结果包含 chunk_id、标题、章节路径、正文、各阶段排名分数。
    引用编号 [n] 对应返回列表第 n 项（1-based）。
    """
    result = await _retriever.retrieve(
        query, mode="hybrid", use_rerank=True, rewrite_mode="none",
        acl=acl, topk=top_k,
    )
    return [
        {
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
        for it in result.items
    ]


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
