from app.indexing.manifest import Manifest


def test_upsert_get_all_remove(tmp_path):
    m = Manifest(str(tmp_path / "m.db"))
    assert m.get("a") is None
    m.upsert("a", path="a.md", md5="h1", chunk_count=3, version="v1",
             valid_from="2026-01-01", valid_until=None)
    got = m.get("a")
    assert got["md5"] == "h1" and got["chunk_count"] == 3
    assert got["valid_from"] == "2026-01-01" and got["valid_until"] is None
    assert set(m.all()) == {"a"}

    m.upsert("a", path="a.md", md5="h2", chunk_count=4, version="v2",
             valid_from="2026-01-01", valid_until="2026-12-31")
    assert m.get("a")["md5"] == "h2" and m.get("a")["chunk_count"] == 4

    m.remove("a")
    assert m.get("a") is None
    assert m.all() == {}


def test_reopen_persists(tmp_path):
    db = str(tmp_path / "m.db")
    Manifest(db).upsert("b", path="b.md", md5="x", chunk_count=1, version="v1",
                        valid_from=None, valid_until=None)
    assert Manifest(db).get("b")["md5"] == "x"
