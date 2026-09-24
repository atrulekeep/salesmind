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
            changed_doc = not record or record["md5"] != doc.md5
            if not changed_doc:
                report.skipped += 1
            elif record:
                report.rebuilt += 1
                await self._vec.delete_doc(doc_id)
            else:
                report.added += 1
            chunks = chunk_document(
                doc, self.s.chunk_size, self.s.chunk_overlap
            )
            all_chunks.extend(chunks)
            if changed_doc:
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
