# SalesMind · 全离线企业销售知识库 RAG Agent

Ollama 本地推理 + 本地嵌入/重排，销售知识库“查资料/对口径”，数据不出域。

## 架构（P0–P1）

corpus(Markdown+frontmatter) → 结构感知分块(chunk_id) →
BM25(jieba) + Chroma(bge-large) 双路召回 → RRF 稳定键融合 →
bge-reranker-v2-m3 精排(可选) → multi-query/HyDE 改写(可选) → qwen3.5 生成（强制引用）

## 快速开始

```bash
ollama pull bge-large:latest && ollama pull qwen3.5:9b
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env
.venv/bin/python -m app.cli build-index --full
.venv/bin/python -m app.cli ask "专业版包年多少钱" --mode hybrid --rerank --rewrite both
```

可选重排依赖：`.venv/bin/pip install -e ".[rerank]"`（未安装时自动降级为不重排）。

## 关键设计

- chunk_id=`{doc_id}#{index}` 稳定键，RRF 不再用 Python 对象 id；
- BM25 用 jieba 中文分词，SKU/数字/百分比精确命中；
- manifest.md5 增量索引：未变跳过，变更只重建该文档；过期政策保留但默认过滤；
- 检索四档开关（vector / hybrid / +rerank / +改写）供评测跑对比曲线。

## 测试

`.venv/bin/pytest -q`（全部为无网络单测；真机验证步骤见 docs/plans）。
