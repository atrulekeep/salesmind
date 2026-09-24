import pytest
from pydantic import ValidationError

from app.config import get_settings
from app.schemas import Chunk


def _chunk(**over):
    base = {
        "chunk_id": "faq#0", "doc_id": "faq", "content": "标题：FAQ\n问题",
        "raw_content": "问题", "chunk_index": 0, "chunk_type": "markdown",
        "title": "FAQ", "section_path": "FAQ", "source": "data/corpus/faq.md",
        "doc_version": "v1", "valid_from": None, "valid_until": None,
        "acl": "public", "content_hash": "abc",
    }
    base.update(over)
    return base


def test_chunk_forbids_extra_fields():
    with pytest.raises(ValidationError):
        Chunk(**_chunk(unknown_field=1))


def test_chunk_default_and_values():
    c = Chunk(**_chunk())
    assert c.chunk_id == "faq#0"
    assert c.acl == "public"
    assert c.valid_until is None


def test_settings_defaults():
    s = get_settings()
    assert s.embedding_dim == 1024
    assert s.rrf_k == 60
    assert s.topk_final == 5
    assert s.chroma_path.endswith("data/index/chroma")
