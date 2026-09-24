# SalesMind P0–P1 检索核心 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建成 SalesMind 全离线 RAG 后端的索引层与检索层（含修复后的 RRF、中文 BM25、Ollama 中文嵌入、reranker、查询改写），CLI 可端到端问答，四档管线开关齐备。

**Architecture:** 去 LangChain 运行时依赖，自有分层：indexing（加载/分块/嵌入/Chroma/BM25/manifest）→ retrieval（改写/双路召回/RRF/重排）→ generation 最小 prompt → cli。Chroma 与 BM25 用稳定 `chunk_id="{doc_id}#{index}"` 对齐；检索阶段全程产出 trace。

**Tech Stack:** Python 3.11（项目内 `.venv`）、chromadb、rank-bm25、jieba、httpx、pydantic v2、pydantic-settings、PyYAML；reranker 可选依赖 transformers + torch；Ollama `bge-large:latest` 嵌入、`qwen3.5:9b` 生成、`BAAI/bge-reranker-v2-m3` 重排。

## Global Constraints

- 工作目录固定为 `/Users/huangyuluo/code/salesmind`，venv 解释器：`/Users/huangyuluo/miniconda3/envs/py311/bin/python`
- 所有模块异步边界：IO（Ollama/Chroma）用 async / `asyncio.to_thread`；RRF/分词/指标保持纯同步函数
- Pydantic v2 模型一律 `model_config = ConfigDict(extra="forbid")`
- 分块长度按**字符**计，默认 512/overlap 64；chunk_id 格式严格为 `{doc_id}#{chunk_index}`
- 外部依赖（Ollama/HF）必须有降级路径；不允许网络依赖进单测（httpx 用 `MockTransport`，索引用 tmp_path + fake）
- 时间一律用 ISO 字符串 `YYYY-MM-DD`；元数据 None 写入 Chroma 前转为空字符串
- 提交纪律：计划含 checkpoint 提交步骤，但**首次 git commit 前必须先征得用户同意**
- Spec：`docs/specs/2026-09-23-salesmind-rag-design.md`（本计划只实现 P0–P1）

---

## File Structure

| 文件 | 职责 |
|---|---|
| `pyproject.toml` | 依赖（含可选 rerank/dev extras）、pytest 配置、可编辑安装 |
| `.env.example` / `.gitignore` | 配置样例 / 忽略规则 |
| `app/__init__.py` 等包初始化 | 包标记 |
| `app/config.py` | pydantic-settings 全局配置（env 前缀 `SALESMIND_`） |
| `app/schemas.py` | `Chunk` / `RetrievedChunk` / `RetrievalTrace` 等 |
| `app/indexing/loader.py` | frontmatter 解析、corpus 扫描、md5 |
| `app/indexing/chunker.py` | 递归切分、Markdown 结构切分、表格保结构、标题注入、编号 |
| `app/indexing/manifest.py` | SQLite 文档版本清单（增量判变） |
| `app/indexing/embeddings.py` | Ollama `/api/embed` 批量封装（重试/维度断言） |
| `app/indexing/vector_store.py` | Chroma 增删查 + where 过滤 |
| `app/indexing/bm25_store.py` | jieba 分词、BM25Okapi、chunks.jsonl 持久化 |
| `app/indexing/pipeline.py` | 增量索引编排（skip/rebuild/remove） |
| `app/retrieval/fuser.py` | 纯函数 RRF |
| `app/llm.py` | Ollama chat（JSON 模式 / 流式事件 / usage） |
| `app/retrieval/rewrite.py` | multi-query + HyDE（1 次调用，失败降级） |
| `app/retrieval/reranker.py` | bge-reranker-v2-m3 懒加载，不可用返回 None |
| `app/retrieval/retriever.py` | 检索管线装配 + ACL/时效过滤 + trace |
| `app/generation/prompt.py` | 编号上下文组装、强制引用/拒答 system prompt |
| `app/cli.py` | `build-index` / `ask` 子命令 |
| `data/corpus/*.md` | 4 篇 P0 种子语料（产品/现行报价/废止报价/FAQ） |
| `tests/test_*.py` | 全部为无网络单测 |

---

## Task 1: 项目骨架与依赖

**Files:**
- Create: `pyproject.toml`, `.env.example`, `.gitignore`, `app/__init__.py`, `app/indexing/__init__.py`, `app/retrieval/__init__.py`, `app/generation/__init__.py`, `tests/__init__.py`, `data/corpus/.gitkeep`, `data/index/.gitkeep`

**Interfaces:**
- Produces: 可导入的 `app` 包；`.venv`；`pytest` 可运行

- [ ] **Step 1: 建目录与虚拟环境**

```bash
mkdir -p /Users/huangyuluo/code/salesmind/{app/indexing,app/retrieval,app/generation,tests,data/corpus,data/index,docs/plans,docs/specs}
cd /Users/huangyuluo/code/salesmind
/Users/huangyuluo/miniconda3/envs/py311/bin/python -m venv .venv
.venv/bin/pip install -U pip
```

- [ ] **Step 2: 写 `pyproject.toml`**

```toml
[project]
name = "salesmind"
version = "0.1.0"
description = "全离线企业销售知识库 RAG Agent"
requires-python = ">=3.11"
dependencies = [
    "chromadb>=0.5.0,<0.7",
    "rank-bm25>=0.2.2",
    "jieba>=0.42.1",
    "httpx>=0.27",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "PyYAML>=6.0",
]

[project.optional-dependencies]
rerank = ["transformers>=4.44", "torch>=2.2"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "ruff>=0.6"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["app*"]
```

- [ ] **Step 3: 写 `.env.example`**

```env
SALESMIND_PROVIDER=ollama
SALESMIND_OLLAMA_BASE_URL=http://127.0.0.1:11434
SALESMIND_LLM_MODEL=qwen3.5:9b
SALESMIND_EMBEDDING_MODEL=bge-large:latest
SALESMIND_EMBEDDING_DIM=1024
SALESMIND_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
SALESMIND_HF_ENDPOINT=https://hf-mirror.com
SALESMIND_CHUNK_SIZE=512
SALESMIND_CHUNK_OVERLAP=64
SALESMIND_TOPN_RECALL=20
SALESMIND_TOPK_FINAL=5
SALESMIND_RRF_K=60
SALESMIND_EMBED_BATCH=16
```

- [ ] **Step 4: 写 `.gitignore`**

```gitignore
.venv/
__pycache__/
.pytest_cache/
*.pyc
.env
data/index/*
!data/index/.gitkeep
.ruff_cache/
```

- [ ] **Step 5: 建空包文件并安装**

```bash
cd /Users/huangyuluo/code/salesmind
for f in app/__init__.py app/indexing/__init__.py app/retrieval/__init__.py app/generation/__init__.py tests/__init__.py; do printf '"""SalesMind."""\n' > "$f"; done
touch data/corpus/.gitkeep data/index/.gitkeep
.venv/bin/pip install -e ".[dev]"
```

- [ ] **Step 6: 验证安装**

Run: `.venv/bin/python -c "import chromadb, jieba, rank_bm25, httpx, pydantic, pydantic_settings, yaml; print('ok')"`
Expected: 输出 `ok`（首次 import chromadb 可能有解压日志，无 Traceback）

- [ ] **Step 7: 冒烟 pytest**

Run: `.venv/bin/pytest -q`
Expected: `no tests ran`（退出码 5 正常）

- [ ] **Step 8: Checkpoint**

`git init`（若尚未初始化），在**征得用户同意后**：`git add -A && git commit -m "chore: scaffold salesmind project"`

---

## Task 2: 配置与核心数据模型

**Files:**
- Create: `app/config.py`, `app/schemas.py`
- Test: `tests/test_schemas_config.py`

**Interfaces:**
- Produces:
  - `Settings`（字段名见代码，env 前缀 `SALESMIND_`），`get_settings() -> Settings`（lru_cache）
  - `Chunk`（字段：chunk_id/doc_id/content/raw_content/chunk_index/chunk_type/title/section_path/source/doc_version/valid_from/valid_until/acl/content_hash）
  - `RetrievedChunk`（chunk + bm25_rank/vector_rank/rrf_score/rerank_score/final_rank）

- [ ] **Step 1: 写失败测试 `tests/test_schemas_config.py`**

```python
from app.schemas import Chunk
from app.config import get_settings
import pytest
from pydantic import ValidationError


def _chunk(**over):
    base = dict(
        chunk_id="faq#0", doc_id="faq", content="标题：FAQ\n问题", raw_content="问题",
        chunk_index=0, chunk_type="markdown", title="FAQ", section_path="FAQ",
        source="data/corpus/faq.md", doc_version="v1", valid_from=None,
        valid_until=None, acl="public", content_hash="abc",
    )
    base.update(over)
    return base


def test_chunk_forbids_extra_fields():
    with pytest.raises(ValidationError):
        Chunk(**_chunk(unknown_field=1))


def test_chunk_default_and_values():
    c = Chunk(**_chunk())
    assert c.chunk_id == "faq#0"
    assert c.acl == "public"
    assert c.valid_until is None


def test_settings_defaults():
    s = get_settings()
    assert s.embedding_dim == 1024
    assert s.rrf_k == 60
    assert s.topk_final == 5
    assert s.chroma_path.endswith("data/index/chroma")
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/pytest tests/test_schemas_config.py -q`
Expected: FAIL（`ModuleNotFoundError: app.config`）

- [ ] **Step 3: 写 `app/config.py`**

```python
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SALESMIND_", extra="ignore")

    provider: str = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3.5:9b"
    embedding_model: str = "bge-large:latest"
    embedding_dim: int = 1024
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    hf_endpoint: str = "https://hf-mirror.com"

    corpus_dir: str = "data/corpus"
    index_dir: str = "data/index"
    chunk_size: int = 512
    chunk_overlap: int = 64
    rrf_k: int = 60
    topn_recall: int = 20
    topk_final: int = 5
    embed_batch: int = 16
    request_timeout: float = 60.0
    default_acl: str = "public"

    @property
    def chroma_path(self) -> str:
        return f"{self.index_dir}/chroma"

    @property
    def chunks_path(self) -> str:
        return f"{self.index_dir}/chunks.jsonl"

    @property
    def manifest_path(self) -> str:
        return f"{self.index_dir}/manifest.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: 写 `app/schemas.py`**

```python
from pydantic import BaseModel, ConfigDict


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    content: str
    raw_content: str
    chunk_index: int
    chunk_type: str  # markdown | table | recursive
    title: str
    section_path: str
    source: str
    doc_version: str
    valid_from: str | None
    valid_until: str | None
    acl: str
    content_hash: str


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    final_rank: int
```

- [ ] **Step 5: 运行确认通过**

Run: `.venv/bin/pytest tests/test_schemas_config.py -q`
Expected: `3 passed`

- [ ] **Step 6: Checkpoint**

`git add app/config.py app/schemas.py tests/test_schemas_config.py && git commit -m "feat: settings and core schemas"`

---

## Task 3: 文档加载器（frontmatter + 扫描 + md5）

**Files:**
- Create: `app/indexing/loader.py`
- Test: `tests/test_loader.py`

**Interfaces:**
- Produces:
  - `parse_document(path: str) -> tuple[dict, str]`（meta 含默认值；body 不含 frontmatter）
  - `scan_corpus(corpus_dir: str) -> list[ScannedDoc]`，`ScannedDoc(path, doc_id, title, body, md5, doc_version, valid_from, valid_until, acl)`
  - `file_md5(path) -> str`

- [ ] **Step 1: 写失败测试**

```python
import pytest
from app.indexing.loader import parse_document, file_md5, scan_corpus


def test_parse_frontmatter(tmp_path):
    p = tmp_path / "pricing.md"
    p.write_text(
        '---\ndoc_id: pricing_2026\ntitle: 报价政策\nversion: "v2"\n'
        "valid_from: 2026-01-01\nacl: public\n---\n# 正文标题\n\n年费 19800。\n",
        encoding="utf-8",
    )
    meta, body = parse_document(str(p))
    assert meta["doc_id"] == "pricing_2026"
    assert meta["doc_version"] == "v2"
    assert meta["valid_from"] == "2026-01-01"
    assert meta["valid_until"] is None
    assert meta["acl"] == "public"
    assert body.startswith("# 正文标题")


def test_parse_without_frontmatter_uses_filename(tmp_path):
    p = tmp_path / "faq.md"
    p.write_text("# FAQ\n\n你好。", encoding="utf-8")
    meta, body = parse_document(str(p))
    assert meta["doc_id"] == "faq"
    assert meta["title"] == "faq"
    assert meta["doc_version"] == "v1"
    assert body.startswith("# FAQ")


def test_md5_stable(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("abc", encoding="utf-8")
    assert file_md5(str(p)) == file_md5(str(p))
    assert len(file_md5(str(p))) == 32


def test_scan_corpus_ignores_non_md_and_requires_doc_id(tmp_path):
    (tmp_path / "a.md").write_text("---\ndoc_id: a\ntitle: A\n---\nbody", encoding="utf-8")
    (tmp_path / "b.txt").write_text("ignore", encoding="utf-8")
    docs = scan_corpus(str(tmp_path))
    assert [d.doc_id for d in docs] == ["a"]
    assert docs[0].md5 and docs[0].body == "body"
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/pytest tests/test_loader.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 写 `app/indexing/loader.py`**

```python
"""Markdown 语料加载：YAML frontmatter + 正文。"""
import hashlib
import os
import re

import yaml
from pydantic import BaseModel, ConfigDict

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.S)


class ScannedDoc(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    doc_id: str
    title: str
    body: str
    md5: str
    doc_version: str
    valid_from: str | None
    valid_until: str | None
    acl: str


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def parse_document(path: str) -> tuple[dict, str]:
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    fallback_id = os.path.splitext(os.path.basename(path))[0]
    m = _FM_RE.match(raw)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        body = m.group(2)
    else:
        meta, body = {}, raw
    meta.setdefault("doc_id", fallback_id)
    meta.setdefault("title", fallback_id)
    return (
        {
            "doc_id": str(meta["doc_id"]),
            "title": str(meta["title"]),
            "doc_version": str(meta.get("version", "v1")),
            "valid_from": meta.get("valid_from"),
            "valid_until": meta.get("valid_until"),
            "acl": str(meta.get("acl", "public")),
        },
        body,
    )


def scan_corpus(corpus_dir: str) -> list[ScannedDoc]:
    docs: list[ScannedDoc] = []
    for name in sorted(os.listdir(corpus_dir)):
        if not name.endswith(".md") or name.startswith("."):
            continue
        path = os.path.join(corpus_dir, name)
        if not os.path.isfile(path):
            continue
        meta, body = parse_document(path)
        docs.append(ScannedDoc(path=path, md5=file_md5(path), body=body, **meta))
    return docs
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/pytest tests/test_loader.py -q`
Expected: `4 passed`

- [ ] **Step 5: Checkpoint**

`git add app/indexing/loader.py tests/test_loader.py && git commit -m "feat: markdown loader with frontmatter and corpus scan"`

---

## Task 4: 结构感知分块器（递归切分/标题/表格/注入/编号）

**Files:**
- Create: `app/indexing/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: `ScannedDoc`（Task 3）
- Produces:
  - `recursive_split(text: str, size: int = 512, overlap: int = 64) -> list[str]`
  - `chunk_document(doc: ScannedDoc, chunk_size=512, chunk_overlap=64) -> list[Chunk]`（index 从 0 连续；`content="标题：{section_path}\n{raw}"`；表格块 chunk_type="table"）

- [ ] **Step 1: 写失败测试**

```python
from app.indexing.loader import ScannedDoc
from app.indexing.chunker import recursive_split, chunk_document


def _doc(body, doc_id="d", **meta_over):
    meta = dict(
        path="data/corpus/d.md", doc_id=doc_id, title="D", body=body, md5="x",
        doc_version="v1", valid_from=None, valid_until=None, acl="public",
    )
    meta.update(meta_over)
    return ScannedDoc(**meta)


def test_recursive_split_respects_size_and_overlap():
    text = "句子。" * 400  # 1200 字符
    pieces = recursive_split(text, size=512, overlap=64)
    assert len(pieces) >= 3
    assert all(len(p) <= 512 for p in pieces)
    # 相邻块存在重叠内容
    assert pieces[0][-20:] in pieces[1]


def test_sections_get_title_path_injection():
    body = "# 产品手册\n\n概述内容。\n\n## 套餐版本\n\n专业版很强。\n"
    chunks = chunk_document(_doc(body))
    assert chunks[0].section_path == "产品手册"
    assert any(c.section_path == "产品手册 > 套餐版本" for c in chunks)
    target = [c for c in chunks if c.section_path == "产品手册 > 套餐版本"][0]
    assert target.content.startswith("标题：产品手册 > 套餐版本\n")
    assert target.chunk_id == "d#" + str(target.chunk_index)


def test_chunk_indices_continuous_and_hash_present():
    body = "# A\n\n" + "内容。" * 300
    chunks = chunk_document(_doc(body))
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(len(c.content_hash) == 32 for c in chunks)
    assert all(c.doc_id == "d" for c in chunks)
    assert all(c.chunk_type in ("markdown", "table") for c in chunks)


def test_table_kept_whole_with_header():
    table = "| SKU | 价格 |\n| --- | --- |\n| P01 | 19800 |\n| P02 | 29800 |\n"
    body = f"# 报价\n\n价格见下。\n\n{table}\n表格之后说明。\n"
    chunks = chunk_document(_doc(body))
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    assert "P01" in table_chunks[0].raw_content and "P02" in table_chunks[0].raw_content
    assert "| SKU | 价格 |" in table_chunks[0].raw_content
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/pytest tests/test_chunker.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 写 `app/indexing/chunker.py`**

```python
"""结构感知分块：Markdown 标题层级 + 表格保结构 + 递归字符切分 + 标题注入。"""
import hashlib
import re

from app.schemas import Chunk
from app.indexing.loader import ScannedDoc

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "，", " ", "")


def recursive_split(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    """按分隔符优先级在 size 内贪心切分；相邻块保留至多 overlap 字符重叠。"""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    pieces: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            cut = -1
            for sep in _SEPARATORS:
                idx = window.rfind(sep)
                if idx > 0:
                    cut = start + idx + len(sep)
                    break
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return pieces


def _iter_segments(body: str) -> list[tuple[str, str, str]]:
    """切出 (section_path, kind, text)：标题推进层级路径，表格独立成段。"""
    segments: list[tuple[str, str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def section() -> str:
        return " > ".join(t for _, t in stack)

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            segments.append((section(), "text", text))
        buf.clear()

    lines = body.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        hm = _HEADER_RE.match(line)
        if hm:
            flush()
            level, title = len(hm.group(1)), hm.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            i += 1
            continue
        if _TABLE_LINE_RE.match(line) and i + 1 < len(lines) and _TABLE_LINE_RE.match(lines[i + 1]):
            flush()
            tbl: list[str] = []
            while i < len(lines) and _TABLE_LINE_RE.match(lines[i]):
                tbl.append(lines[i].strip())
                i += 1
            segments.append((section(), "table", "\n".join(tbl)))
            continue
        buf.append(line)
        i += 1
    flush()
    return segments


def _split_table(text: str, size: int) -> list[str]:
    """表格超长时按行分组，每组都带表头与分隔行。"""
    rows = text.splitlines()
    if len(rows) <= 2 or len(text) <= size:
        return [text]
    header, sep, data = rows[0], rows[1], rows[2:]
    groups: list[str] = []
    cur: list[str] = []
    base_len = len(header) + len(sep)
    for row in data:
        if cur and base_len + sum(len(r) + 1 for r in cur) + len(row) + 1 > size:
            groups.append("\n".join([header, sep, *cur]))
            cur = []
        cur.append(row)
    if cur:
        groups.append("\n".join([header, sep, *cur]))
    return groups


def chunk_document(doc: ScannedDoc, chunk_size: int = 512, chunk_overlap: int = 64) -> list[Chunk]:
    drafts: list[tuple[str, str, str]] = []  # section_path, chunk_type, raw
    for section_path, kind, text in _iter_segments(doc.body):
        path = section_path or doc.title
        if kind == "table":
            for piece in _split_table(text, chunk_size):
                drafts.append((path, "table", piece))
        else:
            for piece in recursive_split(text, chunk_size, chunk_overlap):
                drafts.append((path, "markdown", piece))

    chunks: list[Chunk] = []
    for idx, (section_path, chunk_type, raw) in enumerate(drafts):
        content = f"标题：{section_path}\n{raw}"
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{idx}",
                doc_id=doc.doc_id,
                content=content,
                raw_content=raw,
                chunk_index=idx,
                chunk_type=chunk_type,
                title=doc.title,
                section_path=section_path,
                source=doc.path,
                doc_version=doc.doc_version,
                valid_from=doc.valid_from,
                valid_until=doc.valid_until,
                acl=doc.acl,
                content_hash=hashlib.md5(raw.encode("utf-8")).hexdigest(),
            )
        )
    return chunks
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/pytest tests/test_chunker.py -q`
Expected: `4 passed`

- [ ] **Step 5: Checkpoint**

`git add app/indexing/chunker.py tests/test_chunker.py && git commit -m "feat: structure-aware markdown chunker with table grouping and title injection"`

---

## Task 5: Manifest（SQLite 增量判变）

**Files:**
- Create: `app/indexing/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Produces:
  - `Manifest(db_path)`；方法 `upsert(doc_id, *, path, md5, chunk_count, version, valid_from, valid_until) -> None`
  - `get(doc_id) -> dict | None`；`all() -> dict[str, dict]`；`remove(doc_id) -> None`

- [ ] **Step 1: 写失败测试**

```python
from app.indexing.manifest import Manifest


def test_upsert_get_all_remove(tmp_path):
    m = Manifest(str(tmp_path / "m.db"))
    assert m.get("a") is None
    m.upsert("a", path="a.md", md5="h1", chunk_count=3, version="v1",
             valid_from="2026-01-01", valid_until=None)
    got = m.get("a")
    assert got["md5"] == "h1" and got["chunk_count"] == 3
    assert got["valid_from"] == "2026-01-01" and got["valid_until"] is None
    assert set(m.all()) == {"a"}

    m.upsert("a", path="a.md", md5="h2", chunk_count=4, version="v2",
             valid_from="2026-01-01", valid_until="2026-12-31")
    assert m.get("a")["md5"] == "h2" and m.get("a")["chunk_count"] == 4

    m.remove("a")
    assert m.get("a") is None
    assert m.all() == {}


def test_reopen_persists(tmp_path):
    db = str(tmp_path / "m.db")
    Manifest(db).upsert("b", path="b.md", md5="x", chunk_count=1, version="v1",
                        valid_from=None, valid_until=None)
    assert Manifest(db).get("b")["md5"] == "x"
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/pytest tests/test_manifest.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 写 `app/indexing/manifest.py`**

```python
"""SQLite 文档清单：文件 hash/版本/块数/时效，支撑增量索引。"""
import os
import sqlite3
from datetime import datetime, timezone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    path         TEXT NOT NULL,
    md5          TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL,
    version      TEXT NOT NULL,
    valid_from   TEXT,
    valid_until  TEXT,
    indexed_at   TEXT NOT NULL
);
"""


class Manifest:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def upsert(self, doc_id: str, **fields) -> None:
        self._conn.execute(
            """
            INSERT INTO documents(doc_id, path, md5, chunk_count, version,
                                  valid_from, valid_until, indexed_at)
            VALUES (:doc_id, :path, :md5, :chunk_count, :version,
                    :valid_from, :valid_until, :indexed_at)
            ON CONFLICT(doc_id) DO UPDATE SET
                path=excluded.path, md5=excluded.md5, chunk_count=excluded.chunk_count,
                version=excluded.version, valid_from=excluded.valid_from,
                valid_until=excluded.valid_until, indexed_at=excluded.indexed_at
            """,
            {
                "doc_id": doc_id,
                "path": fields["path"],
                "md5": fields["md5"],
                "chunk_count": fields["chunk_count"],
                "version": fields["version"],
                "valid_from": fields.get("valid_from"),
                "valid_until": fields.get("valid_until"),
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self._conn.commit()

    def get(self, doc_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None

    def all(self) -> dict[str, dict]:
        rows = self._conn.execute("SELECT * FROM documents").fetchall()
        return {r["doc_id"]: dict(r) for r in rows}

    def remove(self, doc_id: str) -> None:
        self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/pytest tests/test_manifest.py -q`
Expected: `2 passed`

- [ ] **Step 5: Checkpoint**

`git add app/indexing/manifest.py tests/test_manifest.py && git commit -m "feat: sqlite manifest for incremental indexing"`

---

## Task 6: Ollama 嵌入客户端

**Files:**
- Create: `app/indexing/embeddings.py`
- Test: `tests/test_embeddings.py`

**Interfaces:**
- Produces:
  - `class OllamaEmbedder`：`__init__(base_url, model, dimension, batch=16, timeout=60, client=None)`
  - `async embed_documents(texts: list[str]) -> list[list[float]]`（分批，失败指数退避重试 2 次）
  - `async embed_query(text: str) -> list[float]`
  - 自定义异常 `EmbeddingError`；维度不符抛 `EmbeddingError`

- [ ] **Step 1: 写失败测试（httpx MockTransport，无网络）**

```python
import json
import httpx
import pytest
from app.indexing.embeddings import OllamaEmbedder, EmbeddingError


def make_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ollama")


async def test_embed_documents_batches_and_asserts_dim():
    seen_inputs = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen_inputs.append(payload["input"])
        return httpx.Response(200, json={"embeddings": [[0.1] * 4 for _ in payload["input"]]})

    emb = OllamaEmbedder("http://ollama", "bge", dimension=4, batch=2, client=make_client(handler))
    out = await emb.embed_documents(["a", "b", "c"])
    assert len(out) == 3 and all(len(v) == 4 for v in out)
    assert seen_inputs == [["a", "b"], ["c"]]


async def test_embed_query_returns_vector():
    def handler(request):
        return httpx.Response(200, json={"embeddings": [[0.2] * 4]})

    emb = OllamaEmbedder("http://ollama", "bge", dimension=4, client=make_client(handler))
    assert await emb.embed_query("q") == [0.2] * 4


async def test_wrong_dimension_raises():
    def handler(request):
        return httpx.Response(200, json={"embeddings": [[0.0, 0.0]]})

    emb = OllamaEmbedder("http://ollama", "bge", dimension=4, client=make_client(handler))
    with pytest.raises(EmbeddingError):
        await emb.embed_query("q")


async def test_retry_then_succeed(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"embeddings": [[1.0] * 4]})

    async def _instant(_):
        return 0

    monkeypatch.setattr("asyncio.sleep", _instant)
    emb = OllamaEmbedder("http://ollama", "bge", dimension=4, client=make_client(handler))
    assert await emb.embed_query("q") == [1.0] * 4
    assert calls["n"] == 2
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/pytest tests/test_embeddings.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 写 `app/indexing/embeddings.py`**

```python
"""Ollama 嵌入客户端：/api/embed 批量调用，重试与维度断言。"""
import asyncio
import json

import httpx


class EmbeddingError(RuntimeError):
    pass


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        dimension: int = 1024,
        batch: int = 16,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ):
        self._url = base_url.rstrip("/") + "/api/embed"
        self._model = model
        self._dim = dimension
        self._batch = batch
        self._timeout = timeout
        self._client = client

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        if self._client is not None:
            await self._client.aclose()

    def _own_client(self) -> httpx.AsyncClient:
        return self._client or httpx.AsyncClient(timeout=self._timeout)

    async def _post(self, client: httpx.AsyncClient, inputs: list[str]) -> list[list[float]]:
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = await client.post(
                    self._url, json={"model": self._model, "input": inputs}
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    if attempt < 2:
                        await asyncio.sleep(0.5 * (2**attempt))
                        continue
                    raise EmbeddingError(f"embed service error: HTTP {resp.status_code}")
                resp.raise_for_status()  # 4xx 参数错误：立即终止，不重试
                vectors = json.loads(resp.content)["embeddings"]
                if any(len(v) != self._dim for v in vectors):
                    raise EmbeddingError(
                        f"embedding dim mismatch: expect {self._dim}, "
                        f"got {[len(v) for v in vectors][:3]}"
                    )
                return vectors
            except httpx.TransportError as e:
                last_err = e
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2**attempt))
        raise EmbeddingError(f"embed request failed after retries: {last_err}")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._own_client()
        try:
            out: list[list[float]] = []
            for i in range(0, len(texts), self._batch):
                out.extend(await self._post(client, texts[i : i + self._batch]))
            return out
        finally:
            if self._client is None:
                await client.aclose()

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/pytest tests/test_embeddings.py -q`
Expected: `4 passed`

- [ ] **Step 5: 真机验证（需 Ollama 运行）**

Run:
```bash
curl -s http://127.0.0.1:11434/api/embeddings -d '{"model":"bge-large:latest","prompt":"专业版年费多少钱"}' | .venv/bin/python -c "import sys,json; print(len(json.load(sys.stdin)['embedding']))"
```
Expected: `1024`

- [ ] **Step 6: Checkpoint**

`git add app/indexing/embeddings.py tests/test_embeddings.py && git commit -m "feat: ollama embed client with batching, retries and dimension guard"`

---

## Task 7: Chroma 向量库

**Files:**
- Create: `app/indexing/vector_store.py`
- Test: `tests/test_vector_store.py`（tmp_path + 假向量，无网络）

**Interfaces:**
- Consumes: `Chunk`
- Produces:
  - `class VectorStore`：`__init__(path, collection="salesmind")`
  - `async upsert(chunks: list[Chunk], embeddings: list[list[float]]) -> None`
  - `async delete_doc(doc_id: str) -> None`
  - `async query(embedding, topn, where=None) -> list[tuple[str, float]]`（按 distance 升序）
  - `metadata_for_chroma(chunk) -> dict`（None→""）

- [ ] **Step 1: 写失败测试**

```python
from app.indexing.vector_store import VectorStore, metadata_for_chroma
from app.schemas import Chunk


def _chunk(cid, doc="d", **meta):
    return Chunk(
        chunk_id=cid, doc_id=doc, content=f"c-{cid}", raw_content=f"c-{cid}",
        chunk_index=int(cid.split("#")[1]), chunk_type="markdown", title="T",
        section_path="T", source=f"{doc}.md", doc_version="v1",
        valid_from=meta.get("valid_from"), valid_until=meta.get("valid_until"),
        acl=meta.get("acl", "public"), content_hash="h",
    )


async def test_upsert_query_delete(tmp_path):
    vs = VectorStore(str(tmp_path / "chroma"))
    chunks = [_chunk("d#0"), _chunk("d#1"), _chunk("d#2")]
    vecs = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]]
    await vs.upsert(chunks, vecs)

    hits = await vs.query([1.0, 0.0, 0.0], topn=2)
    assert [cid for cid, _ in hits] == ["d#0", "d#1"]

    await vs.delete_doc("d")
    assert await vs.query([1.0, 0.0, 0.0], topn=2) == []


async def test_where_filter_acl_and_validity(tmp_path):
    vs = VectorStore(str(tmp_path / "chroma"))
    chunks = [
        _chunk("d#0", acl="public"),
        _chunk("d#1", acl="sales"),
        _chunk("p#0", doc="p", valid_until="2025-12-31"),
        _chunk("p#1", doc="p", valid_until=""),
    ]
    # 注意 chroma 需要与集合维度一致，全部 3 维
    await vs.upsert(chunks, [[1.0, 0, 0]] * 4)
    today = "2026-09-23"
    where = {"$and": [
        {"acl": "public"},
        {"$or": [{"valid_until": ""}, {"valid_until": {"$gte": today}}]},
    ]}
    hits = await vs.query([1.0, 0.0, 0.0], topn=10, where=where)
    assert set(cid for cid, _ in hits) == {"d#0", "p#1"}


def test_metadata_none_becomes_empty_string():
    md = metadata_for_chroma(_chunk("d#0"))
    assert md["valid_until"] == "" and md["chunk_index"] == 0 and md["acl"] == "public"
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/pytest tests/test_vector_store.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 写 `app/indexing/vector_store.py`**

```python
"""Chroma 持久化向量库：增删查；None 元数据转空串。"""
import asyncio

import chromadb

from app.schemas import Chunk

_SCALAR = str | int | float | bool


def metadata_for_chroma(chunk: Chunk) -> dict[str, _SCALAR]:
    md = chunk.model_dump(exclude={"content", "raw_content"})
    return {k: ("" if v is None else v) for k, v in md.items()}


class VectorStore:
    def __init__(self, path: str, collection: str = "salesmind"):
        client = chromadb.PersistentClient(path=path)
        self._col = client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        await asyncio.to_thread(
            self._col.upsert,
            ids=[c.chunk_id for c in chunks],
            documents=[c.content for c in chunks],
            embeddings=embeddings,
            metadatas=[metadata_for_chroma(c) for c in chunks],
        )

    async def delete_doc(self, doc_id: str) -> None:
        await asyncio.to_thread(self._col.delete, where={"doc_id": doc_id})

    async def query(
        self, embedding: list[float], topn: int, where: dict | None = None
    ) -> list[tuple[str, float]]:
        res = await asyncio.to_thread(
            self._col.query,
            query_embeddings=[embedding],
            n_results=topn,
            where=where,
            include=["distances"],
        )
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0]
        return list(zip(ids, (float(d) for d in distances), strict=True))
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/pytest tests/test_vector_store.py -q`
Expected: `3 passed`

- [ ] **Step 5: Checkpoint**

`git add app/indexing/vector_store.py tests/test_vector_store.py && git commit -m "feat: chroma vector store with acl/validity where filters"`

---

## Task 8: BM25 存储（jieba 中文分词）

**Files:**
- Create: `app/indexing/bm25_store.py`
- Test: `tests/test_bm25_store.py`

**Interfaces:**
- Produces:
  - `tokenize(text: str) -> list[str]`（jieba 精确模式 + 小写 + 去空白/停用词）
  - `class BM25Store`：`__init__(chunks_path)`；`build(chunks)`；`load() -> bool`；`save()/load()` 内部使用 chunks.jsonl
  - `search(query, topn, allowed_ids=None) -> list[tuple[str, float]]`（只返回正分）
  - 属性 `chunks: dict[str, Chunk]`

- [ ] **Step 1: 写失败测试**

```python
from app.indexing.bm25_store import BM25Store, tokenize
from app.schemas import Chunk


def _chunk(cid, text):
    return Chunk(
        chunk_id=cid, doc_id=cid.split("#")[0], content=text, raw_content=text,
        chunk_index=int(cid.split("#")[1]), chunk_type="markdown", title="T",
        section_path="T", source="x.md", doc_version="v1", valid_from=None,
        valid_until=None, acl="public", content_hash="h",
    )


def test_jieba_tokenizes_chinese_and_keeps_codes():
    toks = tokenize("专业版 SKU-PRO-01 的年费是 19800 元")
    assert "专业版" in toks
    assert "19800" in toks
    assert "sku-pro-01" in toks
    assert "" not in toks and "的" not in toks


async def test_build_search_persist_filter(tmp_path):
    store = BM25Store(str(tmp_path / "chunks.jsonl"))
    store.build([
        _chunk("p#0", "专业版年费 19800 元，包含销售自动化模块"),
        _chunk("p#1", "企业版支持私有化部署与定制开发"),
        _chunk("p#2", "退款政策七个工作日内无理由退款"),
    ])
    hits = store.search("专业版一年多少钱", topn=2)
    assert hits[0][0] == "p#0"
    assert all(cid != "p#2" for cid, _ in hits)

    again = BM25Store(str(tmp_path / "chunks.jsonl"))
    assert again.load() is True
    assert again.search("私有化部署", topn=1)[0][0] == "p#1"

    filtered = store.search("专业版", topn=5, allowed_ids={"p#2"})
    assert filtered == []
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/pytest tests/test_bm25_store.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 写 `app/indexing/bm25_store.py`**

```python
"""中文 BM25：jieba 分词 + rank-bm25，chunks.jsonl 为权威持久化格式。"""
import json
import os

import numpy as np
from rank_bm25 import BM25Okapi

from app.schemas import Chunk

_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "或", "及", "都", "也", "就", "吗", "呢",
    "啊", "把", "被", "对", "为", "以", "于", "之", "其", "我", "你", "他", "她",
    "它", "们", "这", "那", "个", "有", "无", "不", "没", "请", "问", "怎么",
    "如何", "多少", "可以", "能够", "需要",
}


def tokenize(text: str) -> list[str]:
    import jieba

    return [
        t.strip().lower()
        for t in jieba.lcut(text, cut_all=False)
        if t.strip() and t.strip().lower() not in _STOPWORDS
    ]


class BM25Store:
    def __init__(self, chunks_path: str):
        self._path = chunks_path
        self.chunks: dict[str, Chunk] = {}
        self._bm25: BM25Okapi | None = None
        self._order: list[str] = []

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = {c.chunk_id: c for c in chunks}
        self._order = [c.chunk_id for c in chunks]
        corpus = [tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        self.save()

    def save(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            for cid in self._order:
                f.write(self.chunks[cid].model_dump_json() + "\n")

    def load(self) -> bool:
        if not os.path.exists(self._path):
            return False
        chunks: list[Chunk] = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(Chunk.model_validate_json(line))
        if not chunks:
            return False
        self.build(chunks)
        return True

    def search(
        self, query: str, topn: int, allowed_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        idx = np.argsort(scores)[::-1]
        hits: list[tuple[str, float]] = []
        for i in idx:
            cid = self._order[int(i)]
            if scores[i] <= 0:
                break
            if allowed_ids is not None and cid not in allowed_ids:
                continue
            hits.append((cid, float(scores[i])))
            if len(hits) >= topn:
                break
        return hits
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/pytest tests/test_bm25_store.py -q`
Expected: `2 passed`

- [ ] **Step 5: Checkpoint**

`git add app/indexing/bm25_store.py tests/test_bm25_store.py && git commit -m "feat: jieba-tokenized bm25 store with jsonl persistence"`

---

## Task 9: RRF 融合（纯函数，稳定键）

**Files:**
- Create: `app/retrieval/fuser.py`
- Test: `tests/test_fuser.py`

**Interfaces:**
- Produces:
  - `rrf(rank_lists: list[list[str]], k: int = 60, topn: int | None = None, weights: list[float] | None = None) -> list[tuple[str, float]]`
    默认每路权重 1.0；空列表跳过；按融合分降序。

- [ ] **Step 1: 写失败测试**

```python
import pytest
from app.retrieval.fuser import rrf


def test_same_id_from_two_routes_merges():
    out = rrf([["a", "b"], ["a", "c"]], k=60)
    assert out[0][0] == "a"
    expected = 1 / 61 + 1 / 61
    assert out[0][1] == pytest.approx(expected)


def test_order_and_topn():
    out = rrf([["a", "b", "c"], ["b", "a"]], k=60, topn=2)
    assert [cid for cid, _ in out] == ["a", "b"]
    assert len(out) == 2


def test_empty_lists_and_weights():
    assert rrf([]) == []
    assert rrf([[]]) == []
    # 权重 0 的一路不贡献
    out = rrf([["a"], ["b"]], weights=[1.0, 0.0])
    assert out[0][0] == "a"
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/pytest tests/test_fuser.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 写 `app/retrieval/fuser.py`**

```python
"""Reciprocal Rank Fusion：只看排名、对量纲不敏感；键必须是稳定 chunk_id。"""
from collections.abc import Sequence


def rrf(
    rank_lists: Sequence[Sequence[str]],
    k: int = 60,
    topn: int | None = None,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for li, ranked in enumerate(rank_lists):
        w = 1.0 if weights is None else weights[li]
        if w == 0:
            continue
        for rank, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ordered[:topn] if topn else ordered
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/pytest tests/test_fuser.py -q`
Expected: `3 passed`

- [ ] **Step 5: Checkpoint**

`git add app/retrieval/fuser.py tests/test_fuser.py && git commit -m "feat: reciprocal rank fusion over stable chunk ids"`

---

## Task 10: LLM 客户端（JSON 模式 + 流式 + usage）

**Files:**
- Create: `app/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces:
  - `class OllamaLLM`：`__init__(base_url, model, timeout=60, client=None)`
  - `async chat_json(system: str, user: str, temperature=0.1) -> dict`（format=json，剥 ```json 围栏，解析失败抛 `LLMError`）
  - `async stream(messages: list[dict], temperature=0.3) -> AsyncIterator[dict]`，事件 `{"type":"delta","text":...}` 与 `{"type":"done","usage":{...}}`

- [ ] **Step 1: 写失败测试**

```python
import json
import httpx
import pytest
from app.llm import OllamaLLM, LLMError


def client_with(lines=None, obj=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if obj is not None:
            return httpx.Response(200, json=obj)
        body = ("\n".join(json.dumps(l) for l in lines)).encode()
        return httpx.Response(200, content=body, headers={"content-type": "application/x-ndjson"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ollama")


async def test_chat_json_parses_fenced_content():
    obj = {"message": {"content": '```json\n{"paraphrases": ["x"]}\n```'}}
    llm = OllamaLLM("http://ollama", "q", client=client_with(obj=obj))
    out = await llm.chat_json("sys", "user")
    assert out == {"paraphrases": ["x"]}


async def test_chat_json_invalid_raises():
    obj = {"message": {"content": "not json at all"}}
    llm = OllamaLLM("http://ollama", "q", client=client_with(obj=obj))
    with pytest.raises(LLMError):
        await llm.chat_json("sys", "user")


async def test_stream_emits_deltas_and_done_usage():
    lines = [
        {"message": {"role": "assistant", "content": "你"}, "done": False},
        {"message": {"role": "assistant", "content": "好"}, "done": False},
        {"message": {"content": ""}, "done": True,
         "prompt_eval_count": 100, "eval_count": 2},
    ]
    llm = OllamaLLM("http://ollama", "q", client=client_with(lines=lines))
    events = [
        ev
        async for ev in llm.stream([{"role": "user", "content": "hi"}])
    ]
    assert events[:2] == [
        {"type": "delta", "text": "你"},
        {"type": "delta", "text": "好"},
    ]
    assert events[2]["type"] == "done"
    assert events[2]["usage"] == {"prompt_tokens": 100, "completion_tokens": 2}
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/pytest tests/test_llm.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 写 `app/llm.py`**

```python
"""Ollama chat 客户端：JSON 模式与流式事件；OpenAI 兼容通道后续在此抽象。"""
import json
import re

import httpx


class LLMError(RuntimeError):
    pass


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S | re.I)


def _parse_json_content(content: str) -> dict:
    text = content.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"model did not return valid JSON: {text[:120]}") from e
    if not isinstance(data, dict):
        raise LLMError("json model output must be an object")
    return data


class OllamaLLM:
    def __init__(self, base_url: str, model: str, timeout: float = 60.0,
                 client: httpx.AsyncClient | None = None):
        self._url = base_url.rstrip("/") + "/api/chat"
        self._model = model
        self._timeout = timeout
        self._client = client

    def _own_client(self) -> httpx.AsyncClient:
        return self._client or httpx.AsyncClient(timeout=self._timeout)

    async def chat_json(self, system: str, user: str, temperature: float = 0.1) -> dict:
        client = self._own_client()
        try:
            resp = await client.post(
                self._url,
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": temperature},
                },
            )
            resp.raise_for_status()
            return _parse_json_content(resp.json()["message"]["content"])
        except httpx.HTTPError as e:
            raise LLMError(f"chat_json failed: {e}") from e
        finally:
            if self._client is None:
                await client.aclose()

    async def stream(self, messages: list[dict], temperature: float = 0.3):
        client = self._own_client()
        try:
            req = client.build_request(
                "POST",
                self._url,
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": temperature},
                },
            )
            resp = await client.send(req, stream=True)
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                evt = json.loads(line)
                if evt.get("done"):
                    yield {
                        "type": "done",
                        "usage": {
                            "prompt_tokens": evt.get("prompt_eval_count", 0),
                            "completion_tokens": evt.get("eval_count", 0),
                        },
                    }
                    break
                text = (evt.get("message") or {}).get("content", "")
                if text:
                    yield {"type": "delta", "text": text}
        finally:
            if self._client is None:
                await client.aclose()
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/pytest tests/test_llm.py -q`
Expected: `3 passed`

- [ ] **Step 5: Checkpoint**

`git add app/llm.py tests/test_llm.py && git commit -m "feat: ollama llm client with json mode and streaming usage events"`

---

## Task 11: 查询改写（multi-query + HyDE，失败降级）

**Files:**
- Create: `app/retrieval/rewrite.py`
- Test: `tests/test_rewrite.py`

**Interfaces:**
- Consumes: `OllamaLLM.chat_json`
- Produces:
  - `RewriteOutput(BaseModel)`：`paraphrases: list[str]=[]`、`hyde: str | None=None`、`fell_back: bool=False`、`llm_calls: int=0`
  - `class Rewriter`：`__init__(llm)`；`async rewrite(question, mode: str) -> RewriteOutput`，mode∈none/multi/hyde/both；异常或校验失败返回 `fell_back=True` 的空改写

- [ ] **Step 1: 写失败测试**

```python
from app.retrieval.rewrite import Rewriter, RewriteOutput
from app.llm import LLMError


class FakeLLM:
    def __init__(self, payload=None, raise_exc=False):
        self.payload, self.raise_exc = payload, raise_exc
        self.systems = []

    async def chat_json(self, system, user, temperature=0.1):
        self.systems.append(system)
        if self.raise_exc:
            raise LLMError("boom")
        return self.payload


async def test_none_mode_no_call():
    llm = FakeLLM({"paraphrases": ["x"], "hyde": "y"})
    out = await Rewriter(llm).rewrite("问题", "none")
    assert out.paraphrases == [] and out.hyde is None and out.llm_calls == 0
    assert llm.systems == []


async def test_both_mode_one_call():
    llm = FakeLLM({"paraphrases": ["同义1", "同义2"], "hyde": "假想答案"})
    out = await Rewriter(llm).rewrite("问题", "both")
    assert out.paraphrases == ["同义1", "同义2"]
    assert out.hyde == "假想答案"
    assert out.llm_calls == 1 and len(llm.systems) == 1


async def test_multi_and_hyde_modes():
    out = await Rewriter(FakeLLM({"paraphrases": ["a"]})).rewrite("q", "multi")
    assert out.paraphrases == ["a"] and out.hyde is None
    out = await Rewriter(FakeLLM({"hyde": "h"})).rewrite("q", "hyde")
    assert out.paraphrases == [] and out.hyde == "h"


async def test_failure_falls_back_silently():
    out = await Rewriter(FakeLLM(raise_exc=True)).rewrite("q", "both")
    assert out.fell_back is True and out.paraphrases == [] and out.hyde is None
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/pytest tests/test_rewrite.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 写 `app/retrieval/rewrite.py`**

```python
"""查询改写：multi-query 与 HyDE 合并为一次 JSON 调用；任何失败降级为原问题。"""
from pydantic import BaseModel, ConfigDict, ValidationError

from app.llm import LLMError

_SYSTEM = (
    "你是销售知识库检索查询改写器。只输出 JSON，不要解释。\n"
    "- paraphrases：2 条与原问题同义但措辞不同的中文改写（含口语/别称，如年费≈包年≈年度订阅）\n"
    "- hyde：假设你能根据企业销售资料回答时，可能出现的答案段落（80-150 字，含可能的数字口径）"
)


class RewriteOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paraphrases: list[str] = []
    hyde: str | None = None
    fell_back: bool = False
    llm_calls: int = 0


class _LLMJson(BaseModel):
    model_config = ConfigDict(extra="ignore")

    paraphrases: list[str] = []
    hyde: str | None = None


class Rewriter:
    def __init__(self, llm):
        self._llm = llm

    async def rewrite(self, question: str, mode: str) -> RewriteOutput:
        if mode == "none":
            return RewriteOutput()
        try:
            raw = await self._llm.chat_json(_SYSTEM, f"问题：{question}\nmode={mode}")
            parsed = _LLMJson.model_validate(raw)
            paraphrases = []
            if mode in ("multi", "both"):
                paraphrases = [p.strip() for p in parsed.paraphrases if p.strip()]
            hyde = parsed.hyde.strip() if mode in ("hyde", "both") and parsed.hyde else None
            return RewriteOutput(paraphrases=paraphrases, hyde=hyde, llm_calls=1)
        except (LLMError, ValidationError):
            return RewriteOutput(fell_back=True)
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/pytest tests/test_rewrite.py -q`
Expected: `4 passed`

- [ ] **Step 5: Checkpoint**

`git add app/retrieval/rewrite.py tests/test_rewrite.py && git commit -m "feat: multi-query and hyde rewriter with one-call json and fallback"`

---

## Task 12: Reranker 封装（懒加载 / 不可用降级）

**Files:**
- Create: `app/retrieval/reranker.py`
- Test: `tests/test_reranker.py`

**Interfaces:**
- Produces:
  - `class BGEReranker`：`__init__(model_name, hf_endpoint=None)`
  - `available() -> bool`（尝试惰性加载，失败缓存为 False）
  - `rerank(query, chunks: list[Chunk], topk: int) -> list[tuple[str, float]] | None`（不可用/异常返回 None；按 logit 降序）

- [ ] **Step 1: 写失败测试（无 torch 环境验证降级路径）**

```python
from app.retrieval.reranker import BGEReranker
from app.schemas import Chunk


def _chunk(cid):
    return Chunk(
        chunk_id=cid, doc_id="d", content=cid, raw_content=cid, chunk_index=0,
        chunk_type="markdown", title="T", section_path="T", source="x.md",
        doc_version="v1", valid_from=None, valid_until=None, acl="public",
        content_hash="h",
    )


def test_rerank_returns_none_when_backend_unavailable():
    r = BGEReranker("definitely-not-a-real-model-xyz")
    assert r.rerank("q", [_chunk("d#0")], topk=1) is None
    assert r.available() is False
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/pytest tests/test_reranker.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 写 `app/retrieval/reranker.py`**

```python
"""bge-reranker-v2-m3 本地交叉编码器。transformers/torch 为可选依赖，缺失即降级。"""
import os

from app.schemas import Chunk


class BGEReranker:
    def __init__(self, model_name: str, hf_endpoint: str | None = None):
        self._model_name = model_name
        self._tokenizer = None
        self._model = None
        self._device = "cpu"
        self._tried = False
        self._available = False
        if hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

    def available(self) -> bool:
        if not self._tried:
            self._try_load()
        return self._available

    def _try_load(self) -> None:
        self._tried = True
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name
            ).to(self._device).eval()
            self._available = True
        except Exception:
            # 模型未下载/依赖未装/设备不支持：统一降级为不重排
            self._available = False

    def rerank(
        self, query: str, chunks: list[Chunk], topk: int
    ) -> list[tuple[str, float]] | None:
        if not chunks:
            return []
        if not self.available():
            return None
        import torch

        pairs = [[query, c.content] for c in chunks]
        with torch.no_grad():
            inputs = self._tokenizer(
                pairs, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(self._device)
            logits = self._model(**inputs).logits.squeeze(-1).float().cpu().tolist()
        if isinstance(logits, float):
            logits = [logits]
        ordered = sorted(
            ((chunks[i].chunk_id, float(s)) for i, s in enumerate(logits)),
            key=lambda x: x[1], reverse=True,
        )
        return ordered[:topk]
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/pytest tests/test_reranker.py -q`
Expected: `1 passed`（未装 rerank extras，走 ImportError 降级）

- [ ] **Step 5: 真机安装与验证（P1 收尾，网络受限时跳过且不阻塞）**

```bash
.venv/bin/pip install -e ".[rerank]"
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python -c \
"from app.retrieval.reranker import BGEReranker; from app.schemas import Chunk; \
c=Chunk(chunk_id='d#0',doc_id='d',content='专业版年费19800元',raw_content='x',chunk_index=0,\
chunk_type='markdown',title='T',section_path='T',source='x',doc_version='v1',\
valid_from=None,valid_until=None,acl='public',content_hash='h'); \
r=BGEReranker('BAAI/bge-reranker-v2-m3'); print(r.available(), r.rerank('多少钱',[c],1))"
```
Expected: 首次下载模型（数分钟），输出 `True [('d#0', <float>)]`；若下载失败，后续评测将该曲线标记 skipped。

- [ ] **Step 6: Checkpoint**

`git add app/retrieval/reranker.py tests/test_reranker.py && git commit -m "feat: local bge reranker with lazy load and graceful degradation"`

---

## Task 13: 检索管线装配（双路/RRF/开关/过滤/trace）

**Files:**
- Create: `app/retrieval/retriever.py`
- Test: `tests/test_retriever.py`

**Interfaces:**
- Consumes: embedder / vector_store / bm25 / reranker / rewriter / `rrf`
- Produces:
  - `RetrievalTrace(BaseModel)`：vector_queries/vector_rankings/vector_fused/bm25_ranking/fused/timings_ms/llm_calls/rewrite_fell_back
  - `RetrievalResult(BaseModel)`：items `list[RetrievedChunk]`、trace、rerank_available
  - `class Retriever`：`__init__(embedder, vector_store, bm25, reranker=None, rewriter=None, settings=None)`
  - `async retrieve(question, *, mode="hybrid", use_rerank=False, rewrite_mode="none", acl="public", today=None) -> RetrievalResult`

- [ ] **Step 1: 写失败测试（fakes 注入，无网络）**

```python
import pytest
from app.retrieval.retriever import Retriever
from app.schemas import Chunk
from app.retrieval.rewrite import Rewriter


class FakeEmbedder:
    async def embed_query(self, q):
        return [1.0 if "专业" in q else 0.0, 0.0]


class FakeVector:
    def __init__(self, ids):
        self.ids = ids
        self.last_where = None

    async def query(self, emb, topn, where=None):
        self.last_where = where
        return [(cid, 0.1) for cid in self.ids]


class FakeLLM:
    async def chat_json(self, s, u, temperature=0.1):
        return {"paraphrases": ["专业版包年价格"], "hyde": None}


def _chunk(cid, text, acl="public", valid_until=None):
    return Chunk(
        chunk_id=cid, doc_id=cid.split("#")[0], content=text, raw_content=text,
        chunk_index=int(cid.split("#")[1]), chunk_type="markdown", title="T",
        section_path="T", source="x.md", doc_version="v1", valid_from=None,
        valid_until=valid_until, acl=acl, content_hash="h",
    )


def _bm25(chunks, path, tmp_path, answers):
    from app.indexing.bm25_store import BM25Store
    s = BM25Store(str(tmp_path / "chunks.jsonl"))
    s.build(chunks)
    orig = s.search

    def fake_search(q, topn, allowed_ids=None):
        out = orig(q, topn, allowed_ids)
        return out if out else answers

    s.search = fake_search
    return s


class FakeReranker:
    def __init__(self, order): self.order, self.calls = order, 0
    def available(self): return True
    def rerank(self, q, chunks, topk):
        self.calls += 1
        score = {cid: float(len(self.order) - i) for i, cid in enumerate(self.order)}
        return [(cid, score[cid]) for cid in self.order if cid in {c.chunk_id for c in chunks}][:topk]


async def test_hybrid_fuses_and_filters_expired_and_acl(tmp_path):
    chunks = [
        _chunk("p#0", "专业版年费价格政策", acl="public"),
        _chunk("p#1", "旧版专业版年费", acl="public", valid_until="2025-12-31"),
        _chunk("s#0", "销售内部提成", acl="sales"),
    ]
    bm = _bm25(chunks, "x", tmp_path, answers=[("p#0", 1.0)])
    v = FakeVector(["p#0", "p#1", "s#0"])
    r = Retriever(FakeEmbedder(), v, bm, settings=_settings())
    res = await r.retrieve("专业版年费", mode="hybrid", today="2026-09-23")
    ids = [it.chunk.chunk_id for it in res.items]
    assert "p#0" in ids and "p#1" not in ids and "s#0" not in ids
    assert res.trace.bm25_ranking  # BM25 路被调用
    assert v.last_where is not None


async def test_vector_mode_skips_bm25(tmp_path):
    chunks = [_chunk("p#0", "专业版")]
    bm = _bm25(chunks, "x", tmp_path, answers=[("p#0", 9.0)])
    r = Retriever(FakeEmbedder(), FakeVector(["p#0"]), bm, settings=_settings())
    res = await r.retrieve("专业版", mode="vector")
    assert res.trace.bm25_ranking == []


async def test_rerank_reorders_and_trace(tmp_path):
    chunks = [_chunk("p#0", "专业版年费"), _chunk("p#1", "退款")]
    bm = _bm25(chunks, "x", tmp_path, answers=[("p#0", 1.0), ("p#1", 0.5)])
    fake_rk = FakeReranker(["p#1", "p#0"])
    r = Retriever(FakeEmbedder(), FakeVector(["p#0", "p#1"]), bm,
                  reranker=fake_rk, settings=_settings())
    res = await r.retrieve("退款", mode="hybrid", use_rerank=True, topk=2)
    assert [it.chunk.chunk_id for it in res.items][0] == "p#1"
    assert res.items[0].rerank_score is not None
    assert res.rerank_available is True


async def test_rewrite_expands_vector_queries(tmp_path):
    chunks = [_chunk("p#0", "专业版包年订阅价格")]
    bm = _bm25(chunks, "x", tmp_path, answers=[])
    r = Retriever(FakeEmbedder(), FakeVector(["p#0"]), bm,
                  rewriter=Rewriter(FakeLLM()), settings=_settings())
    res = await r.retrieve("多少钱", mode="vector", rewrite_mode="multi")
    # 原问题 + 1 条含“专业”的改写
    assert len(res.trace.vector_queries) >= 2
    assert res.trace.llm_calls == 1


def _settings():
    from app.config import Settings
    return Settings(rrf_k=60, topn_recall=20, topk_final=5)
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/pytest tests/test_retriever.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 写 `app/retrieval/retriever.py`**

```python
"""检索管线：改写 → 双路召回 → RRF → rerank；ACL/时效全程过滤；产出 trace。"""
import asyncio
import time

from pydantic import BaseModel, ConfigDict

from app.config import Settings, get_settings
from app.retrieval.fuser import rrf
from app.schemas import RetrievedChunk
from app.retrieval.rewrite import Rewriter


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
        return {"$and": [
            {"acl": acl},
            {"$or": [{"valid_until": ""}, {"valid_until": {"$gte": today}}]},
        ]}

    async def retrieve(self, question: str, *, mode: str = "hybrid",
                       use_rerank: bool = False, rewrite_mode: str = "none",
                       acl: str = "public", today: str | None = None,
                       topk: int | None = None) -> RetrievalResult:
        from datetime import date

        today = today or date.today().isoformat()
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
        vec_lists: list[list[str]] = []
        vec_tasks = [self._embed.embed_query(q) for q in vec_queries]
        embeddings = await asyncio.gather(*vec_tasks)
        allowed = self._allowed_ids(acl, today)
        where = self._where(acl, today)
        for emb in embeddings:
            hits = await self._vec.query(emb, topn, where=where)
            ranked = [cid for cid, _ in hits if cid in allowed]
            vec_lists.append(ranked)
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
            candidates = [self._bm.chunks[cid] for cid in trace.fused if cid in self._bm.chunks]
            ranked = await asyncio.to_thread(self._rerank.rerank, question, candidates, topk)
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
```

注意：`Rewriter(None)` 在 `mode="none"` 时不会触达 llm；测试中注入真实 `Rewriter(FakeLLM())`。

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/pytest tests/test_retriever.py -q`
Expected: `4 passed`

- [ ] **Step 5: 全量单测回归**

Run: `.venv/bin/pytest -q`
Expected: 全部 passed（约 30 个测试），无网络调用

- [ ] **Step 6: Checkpoint**

`git add app/retrieval/retriever.py tests/test_retriever.py && git commit -m "feat: retrieval pipeline with hybrid recall, filters, rerank switch and trace"`

---

## Task 14: 增量索引管线 + 种子语料 + CLI

**Files:**
- Create: `app/indexing/pipeline.py`, `app/generation/prompt.py`, `app/cli.py`
- Create: `data/corpus/product_catalog.md`, `data/corpus/pricing_2026.md`, `data/corpus/pricing_2025_deprecated.md`, `data/corpus/sales_faq.md`
- Test: `tests/test_pipeline_incremental.py`, `tests/test_prompt.py`

**Interfaces:**
- Produces:
  - `IndexReport(BaseModel)`：added/skipped/rebuilt/removed/total_chunks
  - `class IndexPipeline`：`__init__(settings, embedder, vector_store, bm25, manifest)`；`async run() -> IndexReport`
  - `build_context(chunks: list[Chunk]) -> str`、`build_messages(question, chunks) -> list[dict]`、`REFUSAL: str`
  - CLI：`python -m app.cli build-index [--full]`、`python -m app.cli ask "问题" [--mode vector|hybrid] [--rerank] [--rewrite none|multi|hyde|both]`

- [ ] **Step 1: 写增量管线失败测试（fake embedder/vector，断言只重嵌入变更文档）**

`tests/test_pipeline_incremental.py`：

```python
import pytest
from app.config import Settings
from app.indexing.pipeline import IndexPipeline
from app.indexing.manifest import Manifest
from app.indexing.bm25_store import BM25Store


class FakeEmbedder:
    def __init__(self): self.embedded_texts = []
    async def embed_documents(self, texts):
        self.embedded_texts.extend(texts)
        return [[0.0] * 3 for _ in texts]


class FakeVector:
    def __init__(self): self.deleted, self.upserted = [], []
    async def delete_doc(self, doc_id): self.deleted.append(doc_id)
    async def upsert(self, chunks, embeddings):
        self.upserted.append((len(chunks), [c.doc_id for c in chunks]))


def _settings(tmp_path):
    return Settings(corpus_dir=str(tmp_path / "corpus"), index_dir=str(tmp_path / "index"))


def _write_corpus(corpus_dir):
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "a.md").write_text("---\ndoc_id: a\ntitle: A\n---\n# A\n\n专业版内容。", encoding="utf-8")
    (corpus_dir / "b.md").write_text("---\ndoc_id: b\ntitle: B\n---\n# B\n\n退款说明。", encoding="utf-8")


async def test_full_build_then_skip_then_rebuild_and_remove(tmp_path):
    _write_corpus(tmp_path / "corpus")
    s = _settings(tmp_path)
    emb, vec = FakeEmbedder(), FakeVector()
    bm, mf = BM25Store(s.chunks_path), Manifest(s.manifest_path)

    pipe = IndexPipeline(s, emb, vec, bm, mf)
    report = await pipe.run()
    assert report.added == 2 and report.skipped == 0 and report.total_chunks >= 2
    first_embed = len(emb.embedded_texts)

    # 第二次：无变更 → 0 嵌入、0 删除
    report2 = await pipe.run()
    assert report2.skipped == 2 and len(emb.embedded_texts) == first_embed
    assert vec.deleted == []

    # 改 a：只重嵌 a，删除仅 a（清空跨轮累计记录后断言本轮行为）
    (tmp_path / "corpus" / "a.md").write_text(
        "---\ndoc_id: a\ntitle: A\n---\n# A\n\n专业版新内容完全不同。", encoding="utf-8"
    )
    vec.deleted.clear()
    report3 = await pipe.run()
    assert report3.rebuilt == 1 and report3.skipped == 1
    assert vec.deleted == ["a"]
    assert len(emb.embedded_texts) > first_embed

    # 删 b
    (tmp_path / "corpus" / "b.md").unlink()
    vec.deleted.clear()
    report4 = await pipe.run()
    assert report4.removed == 1 and vec.deleted == ["b"]
    assert mf.get("b") is None
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/pytest tests/test_pipeline_incremental.py -q`
Expected: FAIL（ModuleNotFoundError: app.indexing.pipeline）

- [ ] **Step 3: 写 `app/indexing/pipeline.py`**

```python
"""增量索引：未变跳过，变更只重建该文档，删除同步清理；BM25 全量重建。"""
import os

from pydantic import BaseModel, ConfigDict

from app.config import Settings
from app.indexing.bm25_store import BM25Store
from app.indexing.chunker import chunk_document
from app.indexing.loader import scan_corpus
from app.indexing.manifest import Manifest
from app.schemas import Chunk


class IndexReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    added: int = 0
    rebuilt: int = 0
    skipped: int = 0
    removed: int = 0
    total_chunks: int = 0


class IndexPipeline:
    def __init__(self, settings: Settings, embedder, vector_store,
                 bm25: BM25Store, manifest: Manifest):
        self.s = settings
        self._embed = embedder
        self._vec = vector_store
        self._bm = bm25
        self._mf = manifest

    async def run(self) -> IndexReport:
        os.makedirs(self.s.index_dir, exist_ok=True)
        scanned = {d.doc_id: d for d in scan_corpus(self.s.corpus_dir)}
        prev = self._mf.all()
        report = IndexReport()

        all_chunks: list[Chunk] = []
        changed: list[Chunk] = []
        for doc_id, doc in scanned.items():
            record = prev.get(doc_id)
            if record and record["md5"] == doc.md5:
                report.skipped += 1
            else:
                if record:
                    report.rebuilt += 1
                else:
                    report.added += 1
                await self._vec.delete_doc(doc_id)
            chunks = chunk_document(
                doc, self.s.chunk_size, self.s.chunk_overlap
            )
            all_chunks.extend(chunks)
            if not record or record["md5"] != doc.md5:
                changed.extend(chunks)
                self._mf.upsert(
                    doc_id, path=doc.path, md5=doc.md5, chunk_count=len(chunks),
                    version=doc.doc_version, valid_from=doc.valid_from,
                    valid_until=doc.valid_until,
                )

        for gone in set(prev) - set(scanned):
            await self._vec.delete_doc(gone)
            self._mf.remove(gone)
            report.removed += 1

        if changed:
            embeddings = await self._embed.embed_documents([c.content for c in changed])
            await self._vec.upsert(changed, embeddings)

        self._bm.build(all_chunks)
        report.total_chunks = len(all_chunks)
        return report
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/pytest tests/test_pipeline_incremental.py -q`
Expected: `1 passed`

- [ ] **Step 5: 写 prompt 模块与测试**

`tests/test_prompt.py`：

```python
from app.generation.prompt import build_context, build_messages, REFUSAL
from app.schemas import Chunk


def _chunk(cid):
    return Chunk(
        chunk_id=cid, doc_id="d", content=f"标题：T\n内容{cid}", raw_content="x",
        chunk_index=0, chunk_type="markdown", title="报价政策", section_path="报价",
        source="pricing.md", doc_version="v2", valid_from="2026-01-01",
        valid_until=None, acl="public", content_hash="h",
    )


def test_context_numbering_and_metadata():
    ctx = build_context([_chunk("p#0"), _chunk("p#1")])
    assert "[1]" in ctx and "[2]" in ctx
    assert "报价政策" in ctx and "2026-01-01" in ctx


def test_messages_contains_refusal_rule():
    msgs = build_messages("多少钱", [_chunk("p#0")])
    assert msgs[0]["role"] == "system"
    assert REFUSAL in msgs[0]["content"]
    assert "多少钱" in msgs[1]["content"]
```

`app/generation/prompt.py`：

```python
"""上下文组装与最小生成约束（引用校验/自纠在 P3 增强）。"""
from app.schemas import Chunk

REFUSAL = "根据现有资料未找到相关信息。"

SYSTEM_PROMPT = f"""你是智策云CRM销售知识库助手。只能依据下方引用资料回答，遵守：
1. 每个结论后用 [编号] 标注来源，编号必须来自提供的引用；
2. 引用资料不足以得出答案时，必须且只能回复：{REFUSAL}
3. 注意资料的生效日期，已过期政策不得作为现行口径；
4. 回答简洁，先给结论再补必要说明，不要编造数字。"""


def build_context(chunks: list[Chunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        date = c.valid_from or "未标注日期"
        blocks.append(f"[{i}] 文档：{c.title}｜章节：{c.section_path}｜生效：{date}\n{c.raw_content}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[Chunk]) -> list[dict]:
    context = build_context(chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"引用资料：\n{context}\n\n问题：{question}"},
    ]
```

Run: `.venv/bin/pytest tests/test_prompt.py -q`
Expected: `2 passed`

- [ ] **Step 6: 写 4 篇种子语料**

`data/corpus/product_catalog.md`：

```markdown
---
doc_id: product_catalog
title: 智策云CRM 产品手册
version: "v3"
valid_from: 2026-01-01
acl: public
---

# 智策云CRM 产品手册

## 产品定位

智策云CRM 是面向 B2B 企业的客户关系管理系统，包含销售自动化、客户管理、商机管道、合同与回款管理四大模块，支持 SaaS 标准版与私有化部署两种交付形态。

## 套餐与 SKU

产品按套餐售卖，每个套餐有唯一 SKU 编号，报价与折扣政策均通过 SKU 引用：

| 套餐 | SKU 编号 | 适用客户 | 核心模块 |
| --- | --- | --- | --- |
| 标准版 | ZC-STD-2026 | 50 人以下销售团队 | 客户管理、商机管道 |
| 专业版 | ZC-PRO-2026 | 50-300 人成长型企业 | 全部四大模块、报表中心 |
| 旗舰版 | ZC-FLG-2026 | 300 人以上集团客户 | 全部模块、开放 API、专属集群 |

## 模块清单

### 销售自动化

支持线索自动分配、跟进提醒、邮件群发与行为打分，帮助销售把时间花在高意向客户上。

### 客户管理与商机管道

客户 360 视图统一沉淀联系人、沟通记录与合同历史；商机管道支持自定义阶段与赢单率预测。

### 合同与回款

合同审批流、回款计划、发票申请线上化；与商机管道联动，签约后自动生成回款节点。
```

`data/corpus/pricing_2026.md`：

```markdown
---
doc_id: pricing_2026
title: 智策云CRM 报价与折扣政策（2026 现行版）
version: "v2026"
valid_from: 2026-01-01
acl: public
---

# 报价与折扣政策（2026 现行版）

## 标准报价（年度订阅/包年）

以下价格为年度订阅价（客户口语中也称包年价、年费），按账号数分档：

| SKU 编号 | 套餐 | 年度订阅价（元/年） |
| --- | --- | --- |
| ZC-STD-2026 | 标准版 | 9800 |
| ZC-PRO-2026 | 专业版 | 19800 |
| ZC-FLG-2026 | 旗舰版 | 39800 |

## 阶梯折扣

按年度订阅合同金额给予阶梯折扣，折扣在标准报价基础上计算：

- 合同金额 2 万以下：无折扣；
- 合同金额 2 万（含）至 5 万：9 折；
- 合同金额 5 万（含）至 10 万：85 折；
- 合同金额 10 万（含）以上：8 折，须销售总监审批。

## 特价审批

低于 8 折的特价申请一律走特价审批流程，须提交客户竞品报价截图与商务测算，由商务负责人终审。口头承诺折扣无效，以系统审批结果为准。

## 续费与增购

老客户续费在到期前 60 天内启动；续费同档位享 95 折。年中增购账号按剩余月份折算，不另享阶梯折扣。
```

`data/corpus/pricing_2025_deprecated.md`：

```markdown
---
doc_id: pricing_2025_deprecated
title: 智策云CRM 报价政策（2025 已废止）
version: "v2025"
valid_from: 2025-01-01
valid_until: 2025-12-31
acl: public
---

# 报价政策（2025 已废止）

> 本政策已于 2025-12-31 废止，自 2026-01-01 起执行《报价与折扣政策（2026 现行版）》。本文件仅作历史归档，不得用于新签合同报价。

## 2025 年标准报价

| SKU 编号 | 套餐 | 年费（元/年） |
| --- | --- | --- |
| ZC-STD-2025 | 标准版 | 12800 |
| ZC-PRO-2025 | 专业版 | 25800 |

## 2025 年折扣口径（已失效）

2025 年专业版统一按 95 折销售，合同金额满 3 万即可申请 9 折。该口径已被 2026 年阶梯折扣政策取代。
```

`data/corpus/sales_faq.md`：

```markdown
---
doc_id: sales_faq
title: 销售常见问题 FAQ
version: "v3"
valid_from: 2026-03-01
acl: public
---

# 销售常见问题 FAQ

## 价格与采购

### 客户问：能不能按月付费？

标准版与专业版仅支持年度订阅（包年）；旗舰版私有化项目可另行约定付款节奏，但 SaaS 部分仍按年计费。

### 客户问：专业版包年包含多少账号？

ZC-PRO-2026 年度订阅价 19800 元包含 30 个账号，超出账号按 600 元/账号/年增购。

### 客户问：报价含税吗？

标准报价均为不含税价，开具增值税专用发票时按 6% 计取税额。

## 实施与售后

### 客户问：签约后多久能上线？

SaaS 标准版 3 个工作日内开通；专业版标准实施周期为 10 个工作日；旗舰版私有化部署实施周期 30 个工作日。

### 客户问：数据安全怎么保障？

SaaS 版数据存储于国内云厂商，传输与落盘均加密；旗舰版支持私有化部署，数据完全保留在客户内网。

### 客户问：支持退款吗？

开通后 7 个工作日内、未使用核心模块导入数据的客户，可申请无理由退款；超过 7 个工作日按已使用月份折算。
```

- [ ] **Step 7: 写 CLI `app/cli.py`**

```python
"""SalesMind 命令行：build-index（增量构建）与 ask（端到端问答）。"""
import argparse
import asyncio
import os

from app.config import get_settings
from app.generation.prompt import build_messages
from app.indexing.pipeline import IndexPipeline
from app.indexing.vector_store import VectorStore
from app.indexing.bm25_store import BM25Store
from app.indexing.manifest import Manifest
from app.indexing.embeddings import OllamaEmbedder
from app.llm import OllamaLLM
from app.retrieval.reranker import BGEReranker
from app.retrieval.retriever import Retriever
from app.retrieval.rewrite import Rewriter


def _build_components():
    s = get_settings()
    embedder = OllamaEmbedder(
        s.ollama_base_url, s.embedding_model, dimension=s.embedding_dim,
        batch=s.embed_batch, timeout=s.request_timeout,
    )
    vs = VectorStore(s.chroma_path)
    bm = BM25Store(s.chunks_path)
    bm.load()
    mf = Manifest(s.manifest_path)
    llm = OllamaLLM(s.ollama_base_url, s.llm_model, timeout=s.request_timeout)
    reranker = BGEReranker(s.reranker_model, s.hf_endpoint)
    return s, embedder, vs, bm, mf, llm, reranker


async def cmd_build(full: bool) -> None:
    s = get_settings()
    if full:
        import shutil

        for target in (s.chroma_path, s.chunks_path, s.manifest_path):
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            elif os.path.exists(target):
                os.remove(target)
    embedder = OllamaEmbedder(
        s.ollama_base_url, s.embedding_model, dimension=s.embedding_dim,
        batch=s.embed_batch, timeout=s.request_timeout,
    )
    vs = VectorStore(s.chroma_path)
    bm = BM25Store(s.chunks_path)
    mf = Manifest(s.manifest_path)
    report = await IndexPipeline(s, embedder, vs, bm, mf).run()
    print(report.model_dump())


async def cmd_ask(question: str, mode: str, use_rerank: bool, rewrite_mode: str) -> None:
    s, embedder, vs, bm, _, llm, reranker = _build_components()
    retriever = Retriever(
        embedder, vs, bm,
        reranker=reranker, rewriter=Rewriter(llm), settings=s,
    )
    result = await retriever.retrieve(
        question, mode=mode, use_rerank=use_rerank, rewrite_mode=rewrite_mode
    )
    print(f"\n检索：{len(result.items)} 块 | rerank={result.rerank_available} "
          f"| 耗时={result.trace.timings_ms['total']:.0f}ms")
    messages = build_messages(question, [it.chunk for it in result.items])
    print("回答：")
    usage = {}
    async for ev in llm.stream(messages):
        if ev["type"] == "delta":
            print(ev["text"], end="", flush=True)
        else:
            usage = ev["usage"]
    print("\n\n引用：")
    for it in result.items:
        print(f"  [{it.final_rank}] {it.chunk.title} > {it.chunk.section_path} "
              f"({it.chunk.chunk_id}) rrf={it.rrf_score:.4f}"
              if it.rrf_score else f"  [{it.final_rank}] {it.chunk.chunk_id}")
    print(f"\ntoken：{usage}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="salesmind")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-index", help="构建/增量更新索引")
    p_build.add_argument("--full", action="store_true", help="清空后全量重建")

    p_ask = sub.add_parser("ask", help="端到端问答")
    p_ask.add_argument("question")
    p_ask.add_argument("--mode", choices=["vector", "hybrid"], default="hybrid")
    p_ask.add_argument("--rerank", action="store_true")
    p_ask.add_argument("--rewrite", choices=["none", "multi", "hyde", "both"], default="none")

    args = parser.parse_args()
    if args.cmd == "build-index":
        asyncio.run(cmd_build(args.full))
    else:
        asyncio.run(cmd_ask(args.question, args.mode, args.rerank, args.rewrite))


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: 全量单测**

Run: `.venv/bin/pytest -q`
Expected: 全部 passed（约 34 个）

- [ ] **Step 9: 真机构建索引（需 Ollama；首次嵌入约 1–3 分钟）**

Run:
```bash
cd /Users/huangyuluo/code/salesmind
.venv/bin/python -m app.cli build-index --full
.venv/bin/python -m app.cli build-index
```
Expected: 首次输出 `{'added': 4, 'rebuilt': 0, 'skipped': 0, 'removed': 0, 'total_chunks': N}`（N 为正整数，预期 8–15）；第二次 `skipped: 4`、added/rebuilt/removed 均 0。

- [ ] **Step 10: 真机问答（四档各一次）**

```bash
.venv/bin/python -m app.cli ask "专业版包年多少钱" --mode vector
.venv/bin/python -m app.cli ask "专业版包年多少钱" --mode hybrid
.venv/bin/python -m app.cli ask "去年专业版多少钱" --mode hybrid
.venv/bin/python -m app.cli ask "公司年假有几天" --mode hybrid --rerank --rewrite both
```

Expected:
- 前两题答案含 `19800` 与 `[n]` 引用；hybrid 命中块应包含 `pricing_2026`；
- “去年”一题不得引用 2025 旧价作为现行口径（应给 2026 价或说明以现行政策为准）；
- 未覆盖题触发固定拒答语；
- 若 reranker 未安装，输出 `rerank=False` 且不报错（降级验证）。

- [ ] **Step 11: Checkpoint**

`git add app/indexing/pipeline.py app/generation/prompt.py app/cli.py data/corpus tests && git commit -m "feat: incremental indexing pipeline, seed corpus, cli end-to-end qa"`

---

## Task 15: README 快速开始与 P0–P1 收官验证

**Files:**
- Create: `README.md`

- [ ] **Step 1: 写 `README.md`（只写已实现内容）**

```markdown
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
```

- [ ] **Step 2: 最终验证清单**

```bash
cd /Users/huangyuluo/code/salesmind
.venv/bin/pytest -q
.venv/bin/python -m app.cli build-index            # 必须 skipped=4
.venv/bin/ruff check app tests || true             # 无未定义名/语法错（警告可接受）
```

- [ ] **Step 3: Checkpoint**

`git add README.md && git commit -m "docs: salesmind p0-p1 readme and quickstart"`

---

## Self-Review 结论（作者已核对）

1. **Spec 覆盖**：P0–P1 全部条目有任务——三个必修缺陷（Task 8 jieba / Task 6 中文嵌入 / Task 9 稳定键）、增量 manifest（Task 5、14）、rerank 与降级（Task 12）、改写（Task 11）、管线开关与 trace（Task 13）、CLI 端到端（Task 14）。评测/API/前端/MCP 属后续计划，不在本计划。
2. **类型一致性**：`Chunk`/`RetrievedChunk` 字段、`rrf` 返回 `list[tuple]`、`BM25Store.search` 与 `Retriever` 消费方式、`OllamaLLM.stream` 事件形态在各任务间一致。
3. **无占位符**：每个代码步骤含完整可运行代码；测试均给出完整断言。
4. **已知执行风险**：Task 7 依赖 chromadb 对 `$and/$or` where 的版本支持（已约束 >=0.5）；Task 14 真机问答的答案文本是概率性的，断言只在 Task 8/13 等单测层保证，人工验证看要点（19800/拒答/不引用旧价）。
