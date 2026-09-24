"""Chroma 持久化向量库：增删查；None 元数据转空串。"""
import asyncio

import chromadb

from app.schemas import Chunk

_SCALAR = str | int | float | bool


def _date_ord(value: str | None) -> int:
    """ISO 日期转整数序 YYYYMMDD 供数值比较；空值=0（长期有效）。"""
    if not value:
        return 0
    return int(value.replace("-", ""))


def metadata_for_chroma(chunk: Chunk) -> dict[str, _SCALAR]:
    md = chunk.model_dump(exclude={"content", "raw_content"})
    md = {k: ("" if v is None else v) for k, v in md.items()}
    # Chroma 的 $gte 只接受数值：日期过滤用整数序
    md["valid_from_ord"] = _date_ord(chunk.valid_from)
    md["valid_until_ord"] = _date_ord(chunk.valid_until)
    return md


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
        )
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0]
        return list(zip(ids, (float(d) for d in distances), strict=True))
