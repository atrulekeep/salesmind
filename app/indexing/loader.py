"""Markdown 语料加载：YAML frontmatter + 正文。"""
import datetime
import hashlib
import os
import re

import yaml
from pydantic import BaseModel, ConfigDict


def _as_date_str(value) -> str | None:
    """YAML 会把 2026-01-01 解析成 date/datetime，统一规范成 ISO 字符串。"""
    if value is None:
        return None
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.strftime("%Y-%m-%d")
    return str(value)

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


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
            "valid_from": _as_date_str(meta.get("valid_from")),
            "valid_until": _as_date_str(meta.get("valid_until")),
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
