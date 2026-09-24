"""中文 BM25：jieba 分词 + rank-bm25，chunks.jsonl 为权威持久化格式。"""
import os
import re

import numpy as np
from rank_bm25 import BM25Okapi

from app.schemas import Chunk

# 原子 token：连字符编号(SKU-PRO-01) / 数字(含小数百分号) / 纯英文词
_ATOM_RE = re.compile(r"[A-Za-z]+(?:[-_][A-Za-z0-9]+)+|\d+(?:\.\d+)?%?|[A-Za-z]+")

_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "或", "及", "都", "也", "就", "吗", "呢",
    "啊", "把", "被", "对", "为", "以", "于", "之", "其", "我", "你", "他", "她",
    "它", "们", "这", "那", "个", "有", "无", "不", "没", "请", "问", "怎么",
    "如何", "多少", "可以", "能够", "需要",
}


def tokenize(text: str) -> list[str]:
    """编号/数字/英文正则原子保留（BM25 的精确实体优势），中文间隙走 jieba。"""
    import jieba

    tokens: list[str] = []
    pos = 0
    for m in _ATOM_RE.finditer(text):
        gap = text[pos:m.start()]
        tokens.extend(t for t in jieba.lcut(gap, cut_all=False) if t.strip())
        tokens.append(m.group().lower())
        pos = m.end()
    tokens.extend(t for t in jieba.lcut(text[pos:], cut_all=False) if t.strip())
    return [t for t in (s.strip() for s in tokens) if t and t not in _STOPWORDS]


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
            f.writelines(self.chunks[cid].model_dump_json() + "\n" for cid in self._order)

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
