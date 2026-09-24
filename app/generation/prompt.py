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
        blocks.append(
            f"[{i}] 文档：{c.title}｜章节：{c.section_path}｜生效：{date}\n{c.raw_content}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[Chunk]) -> list[dict]:
    context = build_context(chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"引用资料：\n{context}\n\n问题：{question}"},
    ]
