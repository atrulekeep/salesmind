from app.generation.prompt import REFUSAL, build_context, build_messages
from app.schemas import Chunk


def _chunk(cid):
    return Chunk(
        chunk_id=cid, doc_id="d", content=f"标题：T\n内容{cid}", raw_content="x",
        chunk_index=0, chunk_type="markdown", title="报价政策", section_path="报价",
        source="pricing.md", doc_version="v2", valid_from="2026-01-01",
        valid_until=None, acl="public", content_hash="h",
    )


def test_context_numbering_and_metadata():
    ctx = build_context([_chunk("p#0"), _chunk("p#1")])
    assert "[1]" in ctx and "[2]" in ctx
    assert "报价政策" in ctx and "2026-01-01" in ctx


def test_messages_contains_refusal_rule():
    msgs = build_messages("多少钱", [_chunk("p#0")])
    assert msgs[0]["role"] == "system"
    assert REFUSAL in msgs[0]["content"]
    assert "多少钱" in msgs[1]["content"]
