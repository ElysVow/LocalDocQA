from __future__ import annotations

import pytest

from localdocqa.models import Document
from localdocqa.splitter import TextSplitter, make_splitter


def _long_text(paragraphs: int = 12) -> str:
    sentence = "本系统采用检索增强生成方案，先把文档切成小块，再根据问题召回最相关的片段。"
    return "\n\n".join(f"第{i + 1}段：{sentence * 3}" for i in range(paragraphs))


def test_short_text_is_single_chunk():
    splitter = TextSplitter(chunk_size=800, chunk_overlap=120)
    assert splitter.split_text("很短的一段话。") == ["很短的一段话。"]


def test_chunks_respect_size_limit():
    splitter = TextSplitter(chunk_size=300, chunk_overlap=50, min_chars=10)
    chunks = splitter.split_text(_long_text())
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    # 允许的最大长度 = chunk_size + chunk_overlap（为上下文重叠预留）
    assert all(len(chunk) <= 350 for chunk in chunks)


def test_overlap_between_neighbours():
    splitter = TextSplitter(chunk_size=200, chunk_overlap=40)
    chunks = splitter.split_text(_long_text(20))
    assert chunks[1][:40] == chunks[0][-40:]


def test_empty_text_returns_empty_list():
    assert TextSplitter().split_text("   \n\n  ") == []


def test_invalid_params():
    with pytest.raises(ValueError):
        TextSplitter(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        TextSplitter(chunk_size=0)


def test_split_documents_keeps_metadata():
    splitter = TextSplitter(chunk_size=200, chunk_overlap=20)
    document = Document(text=_long_text(4), source="/tmp/demo.md", doc_id="abc123", page=2)
    chunks = splitter.split_documents([document])
    assert chunks
    assert {chunk.doc_id for chunk in chunks} == {"abc123"}
    assert {chunk.page for chunk in chunks} == {2}
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.source == "/tmp/demo.md" for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_make_splitter_from_config():
    splitter = make_splitter({"size": 400, "overlap": 80, "min_chars": 5})
    assert splitter.chunk_size == 400
    assert splitter.chunk_overlap == 80
