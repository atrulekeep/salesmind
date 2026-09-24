from app.indexing.loader import file_md5, parse_document, scan_corpus


def test_parse_frontmatter(tmp_path):
    p = tmp_path / "pricing.md"
    p.write_text(
        '---\ndoc_id: pricing_2026\ntitle: 报价政策\nversion: "v2"\n'
        "valid_from: 2026-01-01\nacl: public\n---\n# 正文标题\n\n年费 19800。\n",
        encoding="utf-8",
    )
    meta, body = parse_document(str(p))
    assert meta["doc_id"] == "pricing_2026"
    assert meta["doc_version"] == "v2"
    assert meta["valid_from"] == "2026-01-01"
    assert meta["valid_until"] is None
    assert meta["acl"] == "public"
    assert body.startswith("# 正文标题")


def test_parse_without_frontmatter_uses_filename(tmp_path):
    p = tmp_path / "faq.md"
    p.write_text("# FAQ\n\n你好。", encoding="utf-8")
    meta, body = parse_document(str(p))
    assert meta["doc_id"] == "faq"
    assert meta["title"] == "faq"
    assert meta["doc_version"] == "v1"
    assert body.startswith("# FAQ")


def test_md5_stable(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("abc", encoding="utf-8")
    assert file_md5(str(p)) == file_md5(str(p))
    assert len(file_md5(str(p))) == 32


def test_scan_corpus_ignores_non_md_and_requires_doc_id(tmp_path):
    (tmp_path / "a.md").write_text("---\ndoc_id: a\ntitle: A\n---\nbody", encoding="utf-8")
    (tmp_path / "b.txt").write_text("ignore", encoding="utf-8")
    docs = scan_corpus(str(tmp_path))
    assert [d.doc_id for d in docs] == ["a"]
    assert docs[0].md5 and docs[0].body == "body"
