"""查询改写：multi-query 与 HyDE 合并为一次 JSON 调用；任何失败降级为原问题。"""
from pydantic import BaseModel, ConfigDict, ValidationError

from app.llm import LLMError

_SYSTEM = (
    "你是销售知识库检索查询改写器。只输出 JSON，不要解释。\n"
    "- paraphrases：2 条与原问题同义但措辞不同的中文改写（含口语/别称，如年费≈包年≈年度订阅）\n"
    "- hyde：假设你能根据企业销售资料回答时，可能出现的答案段落（80-150 字，含可能的数字口径）"
)


class RewriteOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paraphrases: list[str] = []
    hyde: str | None = None
    fell_back: bool = False
    llm_calls: int = 0


class _LLMJson(BaseModel):
    model_config = ConfigDict(extra="ignore")

    paraphrases: list[str] = []
    hyde: str | None = None


class Rewriter:
    def __init__(self, llm):
        self._llm = llm

    async def rewrite(self, question: str, mode: str) -> RewriteOutput:
        if mode == "none":
            return RewriteOutput()
        try:
            raw = await self._llm.chat_json(_SYSTEM, f"问题：{question}\nmode={mode}")
            parsed = _LLMJson.model_validate(raw)
            paraphrases = []
            if mode in ("multi", "both"):
                paraphrases = [p.strip() for p in parsed.paraphrases if p.strip()]
            hyde = parsed.hyde.strip() if mode in ("hyde", "both") and parsed.hyde else None
            return RewriteOutput(paraphrases=paraphrases, hyde=hyde, llm_calls=1)
        except (LLMError, ValidationError):
            return RewriteOutput(fell_back=True)
