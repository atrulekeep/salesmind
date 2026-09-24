from app.config import Settings
from app.indexing.bm25_store import BM25Store
from app.indexing.manifest import Manifest
from app.indexing.pipeline import IndexPipeline


class FakeEmbedder:
    def __init__(self):
        self.embedded_texts = []

    async def embed_documents(self, texts):
        self.embedded_texts.extend(texts)
        return [[0.0] * 3 for _ in texts]


class FakeVector:
    def __init__(self):
        self.deleted, self.upserted = [], []

    async def delete_doc(self, doc_id):
        self.deleted.append(doc_id)

    async def upsert(self, chunks, embeddings):
        self.upserted.append((len(chunks), [c.doc_id for c in chunks]))


def _settings(tmp_path):
    return Settings(corpus_dir=str(tmp_path / "corpus"), index_dir=str(tmp_path / "index"))


def _write_corpus(corpus_dir):
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "a.md").write_text(
        "---\ndoc_id: a\ntitle: A\n---\n# A\n\n专业版内容。", encoding="utf-8"
    )
    (corpus_dir / "b.md").write_text(
        "---\ndoc_id: b\ntitle: B\n---\n# B\n\n退款说明。", encoding="utf-8"
    )


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
