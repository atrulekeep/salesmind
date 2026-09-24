"""/api/documents：列表 / 上传 / 删除 / 增量重建。

索引逻辑全部复用 IndexPipeline（业务零拷贝）：上传 = 写 corpus 后增量 run，
删除 = 删文件后增量 run，reindex = 增量 run。写操作共用 index_lock 串行化。
"""
import asyncio
import os

import yaml
from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.indexing.loader import parse_document, scan_corpus

router = APIRouter(prefix="/api/documents")


def _write_file(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _status(rec: dict | None, md5: str) -> str:
    if rec is None:
        return "pending"
    return "synced" if rec["md5"] == md5 else "stale"


def _disk_docs(corpus_dir: str) -> dict:
    return {d.doc_id: d for d in scan_corpus(corpus_dir)} if os.path.isdir(corpus_dir) else {}


@router.get("")
async def list_documents(request: Request):
    state = request.app.state
    prev = state.manifest.all()
    disk = _disk_docs(state.settings.corpus_dir)
    items = []
    for doc_id, doc in disk.items():
        rec = prev.get(doc_id)
        items.append({
            "doc_id": doc_id,
            "title": doc.title,
            "file": os.path.basename(doc.path),
            "md5": doc.md5,
            "version": doc.doc_version,
            "valid_from": doc.valid_from,
            "valid_until": doc.valid_until,
            "chunk_count": rec["chunk_count"] if rec else 0,
            "indexed_at": rec["indexed_at"] if rec else None,
            "status": _status(rec, doc.md5),
        })
    for doc_id, rec in prev.items():  # manifest 有但文件已删
        if doc_id not in disk:
            items.append({
                "doc_id": doc_id,
                "title": os.path.basename(rec["path"]),
                "file": os.path.basename(rec["path"]),
                "md5": rec["md5"],
                "version": rec["version"],
                "valid_from": rec["valid_from"],
                "valid_until": rec["valid_until"],
                "chunk_count": rec["chunk_count"],
                "indexed_at": rec["indexed_at"],
                "status": "missing",
            })
    items.sort(key=lambda x: x["doc_id"])
    return {"items": items, "total_chunks": len(state.bm.chunks)}


@router.post("")
async def upload_document(request: Request, file: UploadFile, overwrite: bool = False):
    state = request.app.state
    name = os.path.basename(file.filename or "")
    if not name.endswith(".md") or name.startswith(".") or len(name) <= len(".md"):
        raise HTTPException(422, "仅支持非隐藏的 .md 文件")
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(422, "文件必须为 UTF-8 编码")

    corpus_dir = state.settings.corpus_dir
    os.makedirs(corpus_dir, exist_ok=True)
    # tmp 放子目录：scan_corpus 只扫文件不进目录，parse fallback 的 doc_id 仍取目标文件名
    tmp_dir = os.path.join(corpus_dir, ".tmp-upload")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, name)
    target = os.path.join(corpus_dir, name)
    exists = os.path.exists(target)
    await asyncio.to_thread(_write_file, tmp_path, text)

    replaced = False
    try:
        try:
            meta, _ = await asyncio.to_thread(parse_document, tmp_path)
        except yaml.YAMLError:
            raise HTTPException(422, "frontmatter 解析失败（YAML 语法错误）")
        doc_id = meta["doc_id"]

        disk = _disk_docs(corpus_dir)
        clash = disk.get(doc_id)
        if clash and clash.path != target:
            raise HTTPException(
                400, f"doc_id「{doc_id}」已被文件 {os.path.basename(clash.path)} 占用"
            )
        rec = state.manifest.get(doc_id)
        if rec and rec["path"] != target:
            raise HTTPException(
                400, f"doc_id「{doc_id}」已被索引记录 {os.path.basename(rec['path'])} 占用"
            )
        if exists and not overwrite:
            raise HTTPException(409, detail={
                "conflict": "exists",
                "doc_id": doc_id,
                "title": meta["title"],
                "version": meta["doc_version"],
                "chunk_count": rec["chunk_count"] if rec else 0,
            })

        os.replace(tmp_path, target)
        replaced = True
        async with state.index_lock:
            report = await state.pipeline.run()
        return {
            "doc_id": doc_id, "overwritten": exists,
            "report": report.model_dump(),
        }
    finally:
        if not replaced and os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, request: Request):
    state = request.app.state
    rec = state.manifest.get(doc_id)
    if rec is None:
        raise HTTPException(404, f"未知文档：{doc_id}")
    if os.path.exists(rec["path"]):
        os.remove(rec["path"])
    async with state.index_lock:
        report = await state.pipeline.run()
    return {"removed": doc_id, "report": report.model_dump()}


@router.post("/reindex")
async def reindex(request: Request):
    state = request.app.state
    async with state.index_lock:
        report = await state.pipeline.run()
    return report.model_dump()
