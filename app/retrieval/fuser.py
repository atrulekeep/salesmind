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
