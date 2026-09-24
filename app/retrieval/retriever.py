"""检索管线：改写 → 双路召回 → RRF → rerank；ACL/时效全程过滤；产出 trace。"""
import asyncio
import time
from datetime import date

from pydantic import BaseModel, ConfigDict

from app.config import Settings, get_settings
from app.retrieval.fuser import rrf
from app.retrieval.rewrite import Rewriter
from app.schemas import RetrievedChunk


class RetrievalTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vector_queries: list[str] = []
    vector_rankings: list[list[str]] = []
    vector_fused: list[str] = []
    bm25_ranking: list[str] = []
    fused: list[str] = []
    timings_ms: dict[str, float] = {}
    llm_calls: int = 0
    rewrite_fell_back: bool = False


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RetrievedChunk]
    trace: RetrievalTrace
    rerank_available: bool


class Retriever:
    def __init__(self, embedder, vector_store, bm25, reranker=None,
                 rewriter=None, settings: Settings | None = None):
        self._embed = embedder
        self._vec = vector_store
        self._bm = bm25
        self._rerank = reranker
        self._rewriter = rewriter or Rewriter(None)  # none 模式不会调用 LLM
        self.s = settings or get_settings()

    def _allowed_ids(self, acl: str, today: str) -> set[str]:
        allowed = set()
        for c in self._bm.chunks.values():
            if c.acl != acl:
                continue
            if c.valid_until and c.valid_until < today:
                continue
            allowed.add(c.chunk_id)
        return allowed

    @staticmethod
    def _where(acl: str, today: str) -> dict:
        # Chroma 的 $gte 只接受数值，时效过滤走整数序（0=长期有效）
        today_ord = int(today.replace("-", ""))
        return {"$and": [
            {"acl": acl},
            {"$or": [{"valid_until_ord": 0}, {"valid_until_ord": {"$gte": today_ord}}]},
        ]}

    async def retrieve(self, question: str, *, mode: str = "hybrid",
                       use_rerank: bool = False, rewrite_mode: str = "none",
                       acl: str = "public", today: str | None = None,
                       topk: int | None = None) -> RetrievalResult:
        # 业务口径的“今天”取本地日期；评测/测试可显式传 today
        today = today or date.today().isoformat()  # noqa: DTZ011
        topk = topk or self.s.topk_final
        topn = self.s.topn_recall
        trace = RetrievalTrace()
        t_all = time.perf_counter()

        # ① 改写
        t0 = time.perf_counter()
        rw = await self._rewriter.rewrite(question, rewrite_mode)
        trace.llm_calls = rw.llm_calls
        trace.rewrite_fell_back = rw.fell_back
        trace.timings_ms["rewrite"] = (time.perf_counter() - t0) * 1000

        # ② 向量路（原问题 + paraphrases + HyDE，路内 RRF）
        t0 = time.perf_counter()
        vec_queries = [question, *rw.paraphrases]
        if rw.hyde:
            vec_queries.append(rw.hyde)
        trace.vector_queries = vec_queries
        embeddings = await asyncio.gather(
            *[self._embed.embed_query(q) for q in vec_queries]
        )
        allowed = self._allowed_ids(acl, today)
        where = self._where(acl, today)
        vec_lists: list[list[str]] = []
        for emb in embeddings:
            hits = await self._vec.query(emb, topn, where=where)
            vec_lists.append([cid for cid, _ in hits if cid in allowed])
        trace.vector_rankings = vec_lists
        vec_fused = [cid for cid, _ in rrf(vec_lists, k=self.s.rrf_k, topn=topn)]
        trace.vector_fused = vec_fused
        trace.timings_ms["vector_recall"] = (time.perf_counter() - t0) * 1000

        # ③ BM25 路（仅 hybrid）
        t0 = time.perf_counter()
        bm_list: list[str] = []
        if mode == "hybrid":
            hits = await asyncio.to_thread(self._bm.search, question, topn, allowed)
            bm_list = [cid for cid, _ in hits]
        trace.bm25_ranking = bm_list
        trace.timings_ms["bm25_recall"] = (time.perf_counter() - t0) * 1000

        # ④ RRF 融合
        rank_lists = [vec_fused] + ([bm_list] if bm_list else [])
        fused = rrf(rank_lists, k=self.s.rrf_k, topn=topn)
        trace.fused = [cid for cid, _ in fused]
        fused_score = dict(fused)

        # ⑤ rerank
        rerank_available = False
        final_order: list[tuple[str, float | None]]
        if use_rerank and self._rerank is not None:
            candidates = [
                self._bm.chunks[cid] for cid in trace.fused if cid in self._bm.chunks
            ]
            ranked = await asyncio.to_thread(
                self._rerank.rerank, question, candidates, topk
            )
            if ranked is None:
                final_order = [(cid, None) for cid in trace.fused[:topk]]
            else:
                rerank_available = True
                final_order = ranked
        else:
            final_order = [(cid, None) for cid in trace.fused[:topk]]

        # ⑥ 组装带排名的结果
        vec_rank_map = {cid: i + 1 for i, cid in enumerate(vec_fused)}
        bm_rank_map = {cid: i + 1 for i, cid in enumerate(bm_list)}
        items: list[RetrievedChunk] = []
        for final_i, (cid, rk_score) in enumerate(final_order):
            chunk = self._bm.chunks.get(cid)
            if chunk is None:
                continue
            items.append(RetrievedChunk(
                chunk=chunk,
                bm25_rank=bm_rank_map.get(cid),
                vector_rank=vec_rank_map.get(cid),
                rrf_score=fused_score.get(cid),
                rerank_score=rk_score,
                final_rank=final_i + 1,
            ))
        trace.timings_ms["total"] = (time.perf_counter() - t_all) * 1000
        return RetrievalResult(items=items, trace=trace, rerank_available=rerank_available)
