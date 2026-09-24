# P3 计划：生成引用/校验/自纠 + FastAPI SSE

> 日期：2026-09-24
> 对应 spec：`docs/specs/2026-09-23-salesmind-rag-design.md` §7（生成三道防线）、§9（API 路由）、§12（测试）、§13（P3）
> 前置：P2 已收口（四曲线报告 `eval/reports/20260924-175809.json`，72 测试绿）

## 范围

**做**：
- `app/generation/citations.py`：`[n]` 提取 / 越界校验 / 编号→chunk_id 映射（eval 复用同一实现）
- `app/generation/answer.py`：流式生成 + 引用校验 + 一次自纠 + 聚合事件（answer/citations/citation_ok/refused/usage/latency）
- `app/api/server.py` + 路由：`/health`、`/api/retrieve`、`/api/chat/stream`（POST SSE：`status → retrieved → answer_delta* → [status(citation_retry) → answer_delta*] → done`，事件带 id）
- pytest（mock LLM，不依赖真机）+ 真机冒烟（uvicorn + curl SSE）

**不做（顺延）**：
- `/api/documents` 系列与 reindex 路由 → P4 前端文档管理页需要时再做（§13 P3 范围不含）
- 多轮会话状态（spec §1.2 非目标）
- Provider 抽象 openai 兼容通道（仅在 llm.py 预留，不实现）

## 关键设计决策

1. **自纠回路用「清空重发」**：首轮 delta 正常外发（保住流式体验）；校验不过 → 发 `status` 事件（`citation_retry`，附原因）→ 前端据此清空已渲染文本 → 第二轮 `answer_delta` 重新流式 → `done.answer` 只含最终答案。理由：自纠是小概率路径，不该为它牺牲首答的流式体验。
2. **引用校验逻辑单源**：`extract_cited_ids` 从 `eval/run_eval.py:72` 提升到 `app/generation/citations.py`，eval 改为 import；`eval/metrics.py` 的 `REFUSAL_PHRASE` 改为复用 `app/generation/prompt.py` 的 `REFUSAL`。改动后重跑全部测试。
3. **SSE 实现用 sse-starlette**：`EventSourceResponse` 原生支持 `event/id/data` 三元组与断连处理；手写 StreamingResponse 也能做但要自己拼帧，收益低。
4. **组件单例**：`create_app()` 工厂 + lifespan 内构建一套组件（embedder/vs/bm/mf/llm/reranker/retriever/generator）挂 `app.state`；BM25 启动时 load（53 块毫秒级）；reranker 维持懒加载。并发安全：检索无共享可变状态，LLM 客户端每请求独立（现 `OllamaLLM._own_client` 已按调用建连）。
5. **拒答判定**：最终答案含 `REFUSAL` 短语 → `refused=true`、`citations=[]`；`citation_ok` 对拒答恒为 true（无引用即无非法引用）。
6. **诚实原则**：`done` 载荷如实带 `citation_ok`、`refused`、`rerank_available`、`rewrite_fell_back`，不做静默修补。

## Task 拆分

### Task 1 引用校验模块 `app/generation/citations.py`
- `extract_nums(answer) -> list[int]`：正则提取全部 `[n]`（含重复）
- `validate_citations(answer, n_chunks) -> bool`：所有 `[n]` ∈ [1, n_chunks]；无 `[n]` 且非拒答 → 不通过（鼓励带引用）——注意：拒答答案没有 `[n]` 是合法的，该判定放 answer.py 结合 refused 处理，citations.py 只管编号合法性
- `map_citations(answer, items) -> list[str]`：现 run_eval 逻辑原样迁移（去重保序、仅合法编号）
- `eval/run_eval.py` 删本地实现改 import；`eval/metrics.py` REFUSAL_PHRASE 改 import
- `tests/test_citations.py`：越界拦截、合法通过、去重保序、非法/空输入

### Task 2 生成管线 `app/generation/answer.py`
- `AnswerGenerator.generate(question, items, rewrite_mode...) -> AsyncIterator[dict]`（事件：`answer_delta` / `status` / `done`）
- 流式收集 → `validate_citations` → 不过则构造自纠 prompt（附原答案 + 非法编号列表）重喂一次 → 仍不过 `citation_ok=false` 照常返回（不静默）
- `refused` 检测：答案含 REFUSAL 短语
- `tests/test_answer.py`：mock LLM（合法引用一次过 / 首轮非法+自纠成功 / 两轮失败标 false / 拒答）

### Task 3 API 骨架：依赖 + server + /health + /api/retrieve
- `pyproject.toml` 新增 `fastapi`、`uvicorn[standard]`、`sse-starlette`
- `app/api/server.py`：`create_app()`，lifespan 构建组件挂 `app.state`；`/health` 返回 ollama/embedder/reranker/索引就绪状态（ollama 用一次轻量 GET 探活，失败不 crash 返回 degraded）
- `/api/retrieve`：POST `{question, mode, use_rerank, rewrite_mode, topk}` → RetrievalResult（items 简化序列化：chunk 元数据 + 各阶段分数 + trace）
- `tests/test_api.py`：TestClient + mock 组件覆盖 /health、/api/retrieve

### Task 4 SSE 聊天端点 `/api/chat/stream`
- POST `{question, mode, use_rerank, rewrite_mode, topk}` → `EventSourceResponse`
- 事件序列（id 单调递增）：`status(retrieving)` → `retrieved(items 简化)` → `answer_delta*` → [`status(citation_retry)` → `answer_delta*`] → `done`
- `done` 载荷：`answer / citations / citation_ok / refused / usage / latency_ms / trace`（trace 精简：timings + fused 排名 + rerank_available + rewrite_fell_back）
- `tests/test_api.py` 补 SSE 流断言（事件顺序、done 载荷完整性、自纠路径事件序列）

### Task 5 真机冒烟 + 同源一致性
- `uvicorn app.api.server:create_app --factory --port 8766` 后台起服务
- curl 验证：`/health`、`/api/retrieve`、`/api/chat/stream` 各一次（1 条可答题 + 1 条应拒答题）
- 核对：API 与 CLI `ask` 同一问题检索结果一致（同源证明）；SSE 事件序列符合设计
- 全量 `pytest` + `ruff check .` 收尾

## 验收

- 72+ 新增测试全绿、ruff 零告警
- 真机 SSE 全链路跑通：流式 delta、引用正确、拒答样式正确
- 自纠路径至少一次真实触发（或明确记录未触发原因）
- 未 commit（等用户指令）
