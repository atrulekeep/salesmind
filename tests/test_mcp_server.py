"""MCP Server 测试：验证工具注册与入参模型。"""
import pytest
from pydantic import ValidationError

from app.mcp_server import SearchInput, mcp


class TestSearchInput:
    def test_defaults(self):
        inp = SearchInput(query="价格")
        assert inp.query == "价格"
        assert inp.top_k == 5
        assert inp.acl == "public"

    def test_custom(self):
        inp = SearchInput(query="折扣", top_k=10, acl="internal")
        assert inp.top_k == 10
        assert inp.acl == "internal"

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            SearchInput(query="x", unknown_field=1)

    def test_top_k_bounds(self):
        with pytest.raises(ValidationError):
            SearchInput(query="x", top_k=0)
        with pytest.raises(ValidationError):
            SearchInput(query="x", top_k=21)


class TestToolRegistration:
    def test_knowledge_search_registered(self):
        tools = mcp._tool_manager.list_tools()
        names = [t.name for t in tools]
        assert "knowledge_search" in names

    def test_tool_is_readonly(self):
        """确认没有写工具。"""
        tools = mcp._tool_manager.list_tools()
        names = [t.name for t in tools]
        for name in names:
            assert "delete" not in name
            assert "upload" not in name
            assert "write" not in name
            assert "update" not in name
