# SalesMind · 全离线企业销售知识库 RAG Agent

Ollama 本地推理 + 本地嵌入/重排，销售知识库"查资料/对口径"，数据不出域。

## 架构

```
corpus(Markdown+frontmatter) → 结构感知分块(chunk_id) →
BM25(jieba) + Chroma(bge-large) 双路召回 → RRF 稳定键融合 →
bge-reranker-v2-m3 精排(可选) → multi-query/HyDE 改写(可选) →
qwen3.5 生成（强制引用 + 引用校验 + 一次自纠 + 拒答）
```

同一检索核心供 **CLI / FastAPI / MCP / 评测** 四处复用，业务逻辑零拷贝。

```
┌─────────────┐   ┌──────────────┐   ┌─────────────┐
│  CLI        │   │  Next.js 前端 │   │ Claude/Cursor│
│  (app.cli)  │   │  (web/)      │   │  (MCP stdio) │
└──────┬──────┘   └──────┬───────┘   └──────┬──────┘
       │                 │ BFF 反代          │
       │                 ▼                 │
       │          ┌──────────────┐         │
       └─────────▶│  FastAPI      │◀────────┘
                  │  (port 8766)  │
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐
                  │  Retriever    │  双路召回 → RRF → rerank
                  │  Generator    │  流式生成 + 引用校验
                  └──────────────┘
```

## 快速开始

```bash
ollama pull bge-large:latest && ollama pull qwen3.5:9b
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env
.venv/bin/python -m app.cli build-index --full
.venv/bin/python -m app.cli ask "专业版包年多少钱" --mode hybrid --rerank --rewrite both
```

可选重排依赖：`.venv/bin/pip install -e ".[rerank]"`（未安装时自动降级为不重排）。

## Web 前端

```bash
cd web && pnpm install
pnpm dev          # 开发（port 3000，BFF 反代 → 127.0.0.1:8766）
```

同时启动后端 API：

```bash
.venv/bin/uvicorn app.api.server:create_app --factory --port 8766
```

页面：
- `/` 问答页：SSE 流式回答、[n] 引用角标（点击展开来源卡片）、拒答样式、检索调试抽屉
- `/documents` 文档管理：上传（409 冲突确认覆盖）、删除、重建索引、状态追踪

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | Ollama/索引/reranker 状态 |
| `/api/retrieve` | POST | 只检索不生成，带各阶段分数 |
| `/api/chat/stream` | POST | SSE 流式问答（检索→生成→引用校验→done） |
| `/api/documents` | GET | 文档列表 + 索引状态 |
| `/api/documents` | POST | 上传 .md（409 冲突 → overwrite=true 覆盖） |
| `/api/documents/{doc_id}` | DELETE | 删除文档及其块 |
| `/api/documents/reindex` | POST | 重建索引 |

检索/问答请求体支持 `acl` 字段（`public` | `internal`）做数据密级过滤。

## MCP Server（Claude Desktop / Cursor）

```json
{
  "mcpServers": {
    "salesmind": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "app.mcp_server"],
      "cwd": "/path/to/salesmind"
    }
  }
}
```

暴露单一只读工具 `knowledge_search(query, top_k, acl)`，返回与 `/api/retrieve` 同构的检索结果。不暴露写工具。

## 关键设计

- chunk_id=`{doc_id}#{index}` 稳定键，RRF 不再用 Python 对象 id；
- BM25 用 jieba 中文分词，SKU/数字/百分比精确命中；
- manifest.md5 增量索引：未变跳过，变更只重建该文档；过期政策保留但默认过滤；
- 检索四档开关（vector / hybrid / +rerank / +改写）供评测跑对比曲线；
- 引用三道防线：prompt 强制 → 程序化校验（编号越界拦截）→ 一次 LLM 自纠；
- 拒答识别：检索无果时固定回复"根据现有资料未找到相关信息。"；
- ACL 密级过滤：public / internal 两级，检索全链路生效（BM25 allowed + Chroma where）。

## 测试

```bash
.venv/bin/pytest -q          # 114 个无网络单测
.venv/bin/ruff check app/ tests/  # lint 零告警
cd web && pnpm build         # 前端 production build
```
