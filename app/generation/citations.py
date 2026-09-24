"""引用编号提取与程序化校验（spec §7 第三道防线），API 与评测共用。"""
import re

from app.schemas import RetrievedChunk

CITATION_RE = re.compile(r"\[(\d+)\]")


def extract_nums(answer: str) -> list[int]:
    """按出现顺序提取答案中全部 [n]（含重复）。"""
    return [int(m.group(1)) for m in CITATION_RE.finditer(answer)]


def citation_problem(answer: str, n_chunks: int) -> str | None:
    """校验失败返回问题描述（供自纠重喂），通过返回 None。

    规则：至少一个 [n]；所有编号 ∈ [1, n_chunks]。
    拒答答案无引用属合法，豁免由调用方（answer.py）处理。
    """
    nums = extract_nums(answer)
    if not nums:
        return "回答中没有任何 [编号] 引用标注"
    bad = sorted({n for n in nums if not 1 <= n <= n_chunks})
    if bad:
        return f"引用编号超出可用范围 1-{n_chunks}：{bad}"
    return None


def map_citations(answer: str, items: list[RetrievedChunk]) -> list[str]:
    """答案中 [n] → items 的 chunk_id（去重保序，仅合法编号）。"""
    out: list[str] = []
    seen: set[str] = set()
    for n in extract_nums(answer):
        if 1 <= n <= len(items):
            cid = items[n - 1].chunk.chunk_id
            if cid not in seen:
                out.append(cid)
                seen.add(cid)
    return out
