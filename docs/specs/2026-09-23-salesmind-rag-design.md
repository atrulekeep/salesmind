# SalesMind · 全离线企业销售知识库 RAG Agent 设计文档

> 日期：2026-09-23（Day6）
> 状态：已评审，待实施
> 对应作战计划：Day6–12（检索链路加深、100 条评测、前端、MCP、增量索引）
> 前身：`/Users/huangyuluo/code/agent-rag-trae`（离线 RAG 原型，非 git 仓库，分块器迁移复用）

---

## 1. 背景与目标

面向销售团队"查资料 / 对口径"价值流（新人翻几百页资料、老销售口径不一）的检索增强问答系统。
全离线运行：Ollama 本地推理 + 本地嵌入/重排，数据不出域。

**本项目要证明的不是"搭过 RAG"，而是三件事：**

1. 检索质量的每一步提升都有**可复现的评测数字**（四曲线 + 100 条明细）；
2. 混合检索、重排、改写、引用约束各解决什么具体失效模式，能讲清取舍与代价；
3. 同一套检索核心被 CLI / FastAPI / MCP / 评测器复用，工程边界清晰。

### 1.1 诚实原则（硬约束）

- 旧 README 的"65%→90%"目前**没有评测支撑**。四曲线跑完后，简历数字**一律以实测为准**；
- 语料为**仿真数据**（虚构公司"智策云 CRM"），README、简历、面试口径中如实说明；
- LLM-as-judge 只做抽样参考，不进回归门禁；门禁只认确定性指标。

### 1.2 非目标（YAGNI，本轮明确不做）

- 用户体系/登录/多租户实现（只在元数据层预留 `acl` 字段与过滤位）；
- 服务端多轮会话状态（v1 请求无状态，历史由前端持有）；
- OCR、语音、多模态；
- 网页爬取入库（旧代码的 WebBaseLoader 不迁移）；
- AgentFlow 式工作流编排（RAG 是直链，不引入 LangGraph）。

---

## 2. 现状盘点与三个必修缺陷

`agent-rag-trae` 已有：结构感知分块器（Markdown 标题 / PDF 按页 / 表格保结构 / 多格式加载）、
BM25+向量双路骨架、RRF 融合雏形、Ollama 流式 CLI。

| # | 缺陷 | 位置 | 后果 |
|---|---|---|---|
| 1 | RRF 用 `id(doc)` 标识文档，两路返回不同 Python 对象 | 旧 `hybrid_retriever.py:111` | **同一文档永远合并不上**，融合退化为拼接 |
| 2 | BM25 分词为 `text.lower().split()`，中文无空格 | 旧 `hybrid_retriever.py:23` | **中文 BM25 路基本失效**，整句当一个词 |
| 3 | 嵌入用英文模型 `all-MiniLM-L6-v2`，语料是中文 | `.env` | 语义召回质量差 |

其他差距：无 rerank、无 query 改写、无评测、无引用校验、无服务化、无销售语料；
依赖 LangChain 0.1.17（`RetrievalQA` 已废弃）；conda `rag` 环境不存在。

---

## 3. 总体架构

### 3.1 技术选型（已确认）

- 语言/环境：Python 3.11，项目内 `.venv`；
- **去 LangChain 运行时依赖**：检索/生成核心自有实现，HTTP 直连 Ollama；仅可选保留少量文档 loader；
- 向量库：chromadb 原生客户端（持久化）；
- 关键词：rank-bm25 + jieba；
- 嵌入：Ollama `bge-large:latest`（已下载，334M，1024 维，512 token）；
- 重排：`bge-reranker-v2-m3`（transformers + torch 本地加载，约 600MB，hf-mirror 下载）；
- 生成：Ollama `qwen3.5:9b`（已下载，支持 tools/JSON，262K 上下文）；
- 服务：FastAPI + httpx（SSE）；MCP：mcp 1.x / FastMCP（stdio）；
- 前端：Next.js 15 + React 19 + shadcn/ui + Tailwind；端口后端 8766；
- 仓库：新建 `/Users/huangyuluo/code/salesmind`，git 初始化后公开。

### 3.2 目录结构

```
salesmind/
├── app/
│   ├── __init__.py
│   ├── config.py              # pydantic-settings，全部参数 .env 化
│   ├── llm.py                 # Provider 抽象（ollama 默认 / openai 兼容可选）
│   ├── schemas.py             # Chunk / RetrievedChunk / Answer / EvalCase 等
│   ├── indexing/
│   │   ├── loader.py          # 文档加载（由旧 chunking_strategy 迁移/瘦身）
│   │   ├── chunker.py         # 结构感知分块 + 标题路径注入
│   │   ├── embeddings.py      # Ollama /api/embed 封装（批量）
│   │   ├── vector_store.py    # Chroma 增删查 + where 过滤
│   │   ├── bm25_store.py      # jieba 分词 + BM25Okapi + 持久化重建
│   │   ├── manifest.py        # SQLite：文档 hash/版本/块数/时效/ACL
│   │   └── pipeline.py        # 构建/增量索引编排
│   ├── retrieval/
│   │   ├── rewrite.py         # multi-query + HyDE（1 次 LLM 调用 JSON 输出）
│   │   ├── fuser.py           # RRF（稳定 chunk_id，纯排名）
│   │   ├── reranker.py        # bge-reranker-v2-m3 懒加载/降级
│   │   └── retriever.py       # 检索管线 + 各档开关 + 阶段 trace
│   ├── generation/
│   │   ├── prompt.py          # 强制引用/拒答/日期上下文模板
│   │   ├── citations.py       # 编号提取与程序化校验
│   │   └── answer.py          # 流式生成 + 一次自纠 + 聚合事件
│   ├── api/
│   │   ├── server.py          # FastAPI 入口
│   │   └── routes.py          # chat(SSE)/retrieve/documents/health
│   ├── mcp_server.py          # stdio MCP：knowledge_search（只读）
│   └── cli.py                 # build-index / ask / reindex
├── eval/
│   ├── cases.jsonl            # 100 条标注（gold 精确到 chunk_id）
│   ├── run_eval.py            # 四曲线 + 分类别 + 成本 + 报告
│   ├── metrics.py             # Recall@k/MRR/nDCG/引用/关键词/拒答
│   └── reports/               # <ts>.json + latest.md
├── data/
│   ├── corpus/                # 仿真语料（公开）
│   └── index/                 # Chroma + BM25 快照 + manifest.db（gitignore）
├── web/                       # Next.js 前端
├── scripts/
├── tests/                     # pytest：不依赖 LLM 的纯函数测试
├── docs/specs/                # 本设计文档
├── pyproject.toml
├── .env.example
└── README.md
```

### 3.3 数据流

```
离线：corpus/ → loader → chunker(chunk_id+元数据+标题注入)
        ├─→ Chroma（Ollama bge-large 向量）
        ├─→ chunks.jsonl → BM25（jieba）
        └─→ manifest.db（hash/版本/时效/ACL）

在线：question
      → rewrite（开关：none/multi/hyde/both）
      → 向量路 ∥ BM25 路（topn=20，ACL/时效过滤）
      → RRF 融合（稳定键）
      → rerank（20→5，开关）
      → prompt 组装（编号+标题+日期）
      → qwen 流式生成 → 引用校验（不过则一次自纠）
      → SSE：status/retrieved/answer_delta/done
```

---

## 4. 仿真语料设计

虚构 B2B SaaS 公司"**智策云 CRM**"销售知识库，Markdown 为主，15–20 篇 / 2–4 万字 / 约 200–300 块。

| 类别 | 内容 | 支撑题型 |
|---|---|---|
| 产品手册 | 模块、版本、套餐 SKU 编号、功能清单 | entity 精确实体 |
| 报价与折扣政策 | 阶梯折扣、特价审批门槛 | entity / multihop |
| 政策版本 | **2025 旧版（标注废止日期）+ 2026 现行版**并存 | temporal 时效陷阱 |
| 销售 FAQ | 客户异议、标准口径 | fact / paraphrase |
| 竞品对比 | 3 家竞品功能/价格对照表（Markdown 表格） | 表格块 / multihop |
| 合同与 SLA / 售后 | 付款条款、实施周期、响应时效 | multihop / 数字精确 |

**写作时刻意埋入的评测素材：**

- 跨文档引用（折扣政策引用产品 SKU；SLA 引用售后等级）；
- 同义表述（"年费 / 包年 / 年度订阅"、"客户 / 甲方 / 采购方"）；
- 低频实体（SKU 编号、百分比、工作日天数）；
- 过期内容（旧折扣率，仅能在 `valid_until` 之前生效）；
- 知识库不覆盖话题（如员工薪资、内部组织架构）→ unanswerable 题。

每篇 frontmatter：`doc_id / title / version / valid_from / valid_until(可空) / acl`。

---

## 5. 索引层设计

### 5.1 Chunk 身份与元数据

```
chunk_id     = f"{doc_id}#{chunk_index}"          # 稳定主键，评测 gold 直接引用
content_hash = md5(正文)                           # 增量判变
metadata = {
  doc_id, chunk_id, chunk_index, chunk_type,
  title, section_path,        # H1 > H2 > H3 路径
  source, doc_version,
  valid_from, valid_until,    # 时效过滤；None 表示长期有效
  acl,                        # 默认 "public"
}
```

- **标题路径注入**：每块正文前置 `标题：{section_path}`，块脱离原文仍自包含；
- Markdown 按标题层级切、表格整块保结构（迁移现有正则方案）、PDF/文本递归切；
- 长度按**字符计**（bge 中文约 1 字≈1 token，默认 512 字符 / overlap 64，不超嵌入模型上限）。

### 5.2 增量索引（面试题 Q14）

`manifest.db`：`documents(doc_id, path, md5, chunk_count, version, valid_from, valid_until, indexed_at)`。

启动/手动 reindex 时：

1. 扫描 corpus，md5 未变 → skip；
2. 变更/新增 → 删该 doc 全部块（Chroma by doc_id + 内存清单移除）→ 重新分块嵌入写入；
3. 删除文件 → 清块 + manifest 标记；
4. BM25 为全量内存索引，随块清单整体重建（<1 万块毫秒级）；
5. 旧版政策**保留不删**，查询默认过滤 `valid_until < today`。

### 5.3 嵌入

Ollama `POST /api/embed`，批量调用，失败重试 2 次（指数退避）；
入库时做 embedding 维度断言（1024），防止模型被换后静默写错库。

---

## 6. 检索管线设计

### 6.1 阶段与开关

| 开关 | 取值 | 说明 |
|---|---|---|
| `RETRIEVAL_MODE` | `vector` / `hybrid` | 基线只走向量；hybrid 开 BM25+RRF |
| `USE_RERANK` | bool | bge-reranker-v2-m3 精排 |
| `REWRITE_MODE` | `none` / `multi` / `hyde` / `both` | 查询改写 |
| `TOPN_RECALL` | 默认 20 | 每路召回数 |
| `TOPK_FINAL` | 默认 5 | 精排后返回数 |

四曲线配置固定为：

```
A: vector + 无改写 + 无rerank
B: hybrid(RRF) + 无改写 + 无rerank
C: hybrid + rerank
D: hybrid + rerank + both(multi-query + HyDE)
```

### 6.2 RRF（修好稳定键）

```python
def rrf(rank_lists: list[list[str]], k: int = 60, topn: int = 20):
    scores: dict[str, float] = {}
    for ranked in rank_lists:
        for rank, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:topn]
```

- 键一律用 `chunk_id`，杜绝 `id(obj)`；
- 纯排名、无参、对量纲不敏感；默认两路权重 1:1；
- 权重参数保留（`fusion_weights`），但**只允许被评测数据修改**，不手调。

### 6.3 双路召回

- **BM25 路**：jieba 精确模式分词（去停用词表内置一版常用中文停用词）；对 SKU/编号/百分比天然强；
- **向量路**：原问题 + 每个改写变体分别召回，**路内先 RRF 合并**，再与 BM25 路 RRF；
- HyDE 文本只走向量路（假想答案与目标块语义近，不适合关键词路）；
- 过滤：ACL（Chroma `where` + BM25 后置）、时效（默认滤掉过期块）。

### 6.4 查询改写（成本控制在 1 次 LLM 调用）

一次 JSON-mode 调用同时产出：

```json
{"paraphrases": ["改写1", "改写2"], "hyde": "假想答案一段话"}
```

温度 0.1，schema 用 Pydantic 校验；解析失败降级为只用原问题（记录到 trace，不中断问答）。

### 6.5 Rerank 与降级

- `bge-reranker-v2-m3`，transformers 直接加载（不经过 Ollama，Ollama 不提供 rerank）；
- 懒加载：首次查询时加载，进程内缓存；
- 模型缺失/加载失败：记 warning，跳过精排并在响应里标 `rerank_available=false`；
- 评测跑到该曲线时若模型不可用 → 该曲线**明确标记 skipped**，不静默用无 rerank 结果冒充。

### 6.6 阶段 trace

每次检索内部返回各阶段排名与耗时：

```python
RetrievedChunk(chunk_id, title, source, section_path, valid_from,
               bm25_rank, vector_rank, rrf_score, rerank_score, final_rank)
```

供 API 调试抽屉、评测报告、前端展示复用。

---

## 7. 生成层设计

三道防线：

1. **检索质量**：双路 + RRF + rerank；
2. **Prompt 约束**：上下文按 `[1]…[n]` 编号，每条附标题/来源/生效日期；要求每个结论后标 `[n]`；资料不足时**只允许**回复固定拒答语：`根据现有资料未找到相关信息。`
3. **程序化校验**：正则提取答案中所有 `[n]`，必须 ⊆ 已给编号；
   - 不通过 → 带错误信息重喂**一次**令其自修；
   - 仍不通过 → 正常返回但标 `citation_ok=false`（不静默、不伪造）。

- 流式：Ollama `POST /api/chat`（`stream=true`），httpx 透传 SSE；
- 拒答检测：答案命中固定拒答语 → `refused=true`，不附假引用；
- token/延迟：从 Ollama 响应取 `prompt_eval_count / eval_count`，进 trace 与评测成本统计。

---

## 8. 评测体系（王牌证据）

### 8.1 案例格式（`eval/cases.jsonl`）

```json
{
  "id": "eval-001",
  "category": "entity",
  "question": "专业版的年费是多少？",
  "relevant_chunk_ids": ["pricing_2026#3"],
  "answer_keywords": ["19800"],
  "should_refuse": false
}
```

100 条，语料与题目同源设计（写语料时按知识点清单组织，题目由指定块反向出题）：

| 类别 | 数量 | 考点 |
|---|---|---|
| fact 单点事实 | 30 | 基础召回 |
| entity 精确实体 | 20 | BM25 不可替代性 |
| paraphrase 释义改写 | 20 | 向量/HyDE |
| multihop 跨文档多跳 | 15 | topk 完整性 |
| temporal 时效版本 | 10 | 元数据过滤 |
| unanswerable 无答案 | 5 | 拒答 |

每题至少 1 个、一般 1–3 个 gold chunk；含必要关键词；语料/题目版本锁定（corpus v1 ↔ eval v1）。

### 8.2 指标

- **检索层**：Recall@5、MRR、nDCG@5（二值相关），整体 + 六类别分组；
- **生成层**：
  - 引用精确率 = |引用 ∩ gold| / |引用|；引用召回率 = |引用 ∩ gold| / |gold|；
  - 关键词命中率（必需关键词全出现的比例）；
  - 拒答正确率（unanswerable 拒答 + 可答题未误拒）；
- **成本层**：平均延迟、平均 LLM 调用数、平均 prompt/completion tokens（每曲线一份）；
- **LLM-as-judge（仅参考）**：抽 20 条开放题，本地 qwen 两遍独立打分，报双评一致性；不进门禁。

### 8.3 运行与产物

- `run_eval.py`：asyncio + Semaphore 并发；同一批 cases 依次跑 A→D 四档；
- 产物：
  - `eval/reports/<ts>.json`：100 条×四曲线完整明细；
  - `latest.md`：指标 × 四曲线 × 类别对比表 + 成本表；
- 门禁：阈值检查脚本（指标较基线回退则 exit 1），确定性指标、零 LLM 成本、可进 CI；
- 附加消融：256 / 512 / 1024 三种分块大小的小对比，数据化回答"为什么 512"。

---

## 9. FastAPI 服务层

同一检索核心喂 CLI / API / MCP / 评测，业务逻辑零拷贝。

| 路由 | 方法 | 说明 |
|---|---|---|
| `/api/chat/stream` | POST SSE | `status → retrieved(可选) → answer_delta → done` |
| `/api/retrieve` | POST | 只检索不生成，带各阶段分数（调试/评测复用） |
| `/api/documents` | GET/POST/DELETE | 列表（版本/hash/块数/时效）/ 上传 / 删除 |
| `/api/documents/reindex` | POST | 增量重建，返回 rebuilt/skipped/removed |
| `/health` | GET | Ollama / 嵌入 / reranker / 索引就绪状态 |

- SSE 事件带 `id`（为 Last-Event-ID 续传预留）；
- `done` 载荷：`answer / citations / citation_ok / refused / usage / latency_ms / trace`；
- Provider：`SALESMIND_PROVIDER=ollama|openai`，默认 ollama；openai 兼容端点仅作质量对照，不公开部署。

---

## 10. Next.js 前端（Day11，≤1.5 天）

- **问答页**：消息流 + 流式打字；`[n]` 渲染为角标，点击展开来源卡片（文档/章节/日期/块原文）；拒答独立样式；
- **检索调试抽屉**（演示用）：双路命中、RRF 排名、rerank 前后顺序、阶段耗时——把四曲线优化可视化；
- **文档管理页**：上传 / 列表 / 删除 / 重建索引，显示 hash、版本、增量状态；
- Route Handler 反代后端 SSE（BFF 形态，避免跨域）；
- 明确不做：登录、多用户、富文本、协作。

---

## 11. MCP Server（Day12）

stdio FastMCP，单一只读工具：

```python
@mcp.tool()
def knowledge_search(query: str, top_k: int = 5) -> list[dict]:
    """检索智策云CRM销售知识库，返回要点与引用来源（只读）"""
```

- 入参 Pydantic、`extra="forbid"`；返回与 `/api/retrieve` 同构（同一函数）；
- Claude Desktop / Cursor 实测挂载，截图入 README；
- 不暴露写工具（写操作走 HTTP UI，未来写工具必须 HITL）。

---

## 12. 测试与验收

**pytest（不依赖 LLM，秒级）：**

- RRF：稳定键合并、排名公式、topn；
- jieba 中文分词命中；
- 引用校验：编号越界拦截、合法集合通过、自纠触发条件；
- 增量索引：改单文件只重建该文档、删除文件清块、未变跳过；
- 时效过滤：旧政策默认不命中；
- 改写 JSON 解析失败降级。

**验收物：**

1. 四曲线评测报告（JSON 明细 + Markdown 对比），数字真实；
2. CLI / API / MCP 三处同源结果一致性手工验证；
3. Next.js 页面流式问答 + 引用展开 + 文档管理可演示；
4. Claude Desktop MCP 实测截图；
5. README（架构图、四曲线表、演示脚本）+ 简历数据同步 + 3 分钟录屏。

---

## 13. 实施顺序（对齐 Day6–12）

| 阶段 | 时间 | 内容 | 止损点 |
|---|---|---|---|
| P0 | Day6 | 仓库/venv/配置；chunk_id+元数据；分块迁移；jieba BM25；Chroma+Ollama 嵌入；修好的 RRF；manifest 增量；CLI 问答跑通 | **P0 完成即今日止损点** |
| P1 | Day6（顺延则 Day7） | reranker 接入；改写模块；管线开关；阶段 trace | 模型下载失败不阻塞 P2 |
| P2 | Day7–10 | 语料补全 → 100 题标注 → metrics/runner → 四曲线报告 → 按数据返工参数 | 评测报告是最高优先交付 |
| P3 | Day10–11 | 生成引用/校验/拒答 → FastAPI SSE | — |
| P4 | Day11 | Next.js 问答页/调试抽屉/文档管理 | 可砍文档管理页动画之外的一切美化 |
| P5 | Day12 | MCP、ACL 过滤、README、简历同步、录屏 | — |

---

## 14. 风险与对策

| 风险 | 对策 |
|---|---|
| reranker 模型下载受限 | `HF_ENDPOINT=https://hf-mirror.com`；失败则该曲线标记 skipped，管线降级可跑 |
| qwen3.5:9b 本地生成慢 | 评测并发执行；报告如实记录延迟；生成通道可切 openai 兼容端点做对照 |
| Day6 工程量超预期 | P0 为硬止损点；P1 顺延不影响评测主线 |
| 仿真语料被面试官质疑 | 主动说明"仿真但内部自洽、可公开复现"，评测方法论与指标实现才是重点 |
| 评测数字不如旧宣传 | 以实测为准；讲方法论与各层增益归因，不背数字包袱 |
| 嵌入模型 512 上限 | 分块按字符控制 ≤512；入库断言，超长块强制递归切分 |

---

## 15. 面试讲法锚点（与 04 题库对应）

- Q10 结构感知分块 → `app/indexing/chunker.py` + 256/512/1024 消融数据；
- Q11 混合检索 + RRF → `fuser.py`（稳定键是加分细节）+ 曲线 B 增益；
- Q12 评测 → `eval/` 100 条 + 四曲线 + 确定性门禁，LLM-judge 只抽样；
- Q13 引用/拒答 → 三道防线 + 程序化校验 + 一次自纠；
- Q14 增量更新 → manifest hash + 旧版保留降权；
- Q17 降级 → reranker/Ollama 不可用时的已验证降级路径。
