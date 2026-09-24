"""结构感知分块：Markdown 标题层级 + 表格保结构 + 递归字符切分 + 标题注入。"""
import hashlib
import re

from app.indexing.loader import ScannedDoc
from app.schemas import Chunk

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
