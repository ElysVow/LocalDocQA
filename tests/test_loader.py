from __future__ import annotations

import json

import pytest

from localdocqa.loader import load_file, read_text_file, resolve_inputs


def test_load_markdown(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# 标题\n\n这是正文内容。\n", encoding="utf-8")
    documents = load_file(path)
    assert len(documents) == 1
    assert "这是正文内容" in documents[0].text
    assert documents[0].meta["sha256"]
    assert documents[0].doc_id


def test_load_gbk_text(tmp_path):
    path = tmp_path / "gbk.txt"
    path.write_bytes("中文编码测试：段落一。".encode("gbk"))
    text = read_text_file(path)
    assert "中文编码测试" in text


def test_load_html_strips_tags(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><style>body{}</style></head><body><h1>标题</h1><p>第一段。</p>"
        "<script>alert(1)</script></body></html>",
        encoding="utf-8",
    )
    documents = load_file(path)
    assert "标题" in documents[0].text
    assert "<" not in documents[0].text
    assert "alert" not in documents[0].text


def test_load_csv(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("课程,学分\n数据结构,4\n计算机网络,3\n", encoding="utf-8")
    documents = load_file(path)
    text = documents[0].text
    assert "第1行" in text and "数据结构" in text
    assert "学分: 3" in text


def test_load_json_flatten(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"模型": {"名称": "qwen", "参数": "7B"}}), encoding="utf-8")
    documents = load_file(path)
    assert "模型.名称: qwen" in documents[0].text


def test_unsupported_extension(tmp_path):
    path = tmp_path / "binary.exe"
    path.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError):
        load_file(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_file(tmp_path / "nope.md")


def test_resolve_inputs_filters_and_dedupes(tmp_path):
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "c.exe").write_bytes(b"x")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "d.md").write_text("d", encoding="utf-8")
    hidden = tmp_path / ".hidden.md"
    hidden.write_text("h", encoding="utf-8")

    files = resolve_inputs([str(tmp_path)])
    names = [path.name for path in files]
    assert names == ["a.md", "b.txt", "d.md"]  # 排除 .exe 和隐藏文件
    assert len(resolve_inputs([str(tmp_path), str(tmp_path / "a.md")])) == 3
