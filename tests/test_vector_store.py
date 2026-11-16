from __future__ import annotations

import numpy as np
import pytest

from localdocqa.embeddings import HashingEmbeddings
from localdocqa.models import Chunk
from localdocqa.vector_store import CollectionManager, VectorStore, VectorStoreError


def _build_store(tmp_path, texts, dim=256, name="demo"):
    embedder = HashingEmbeddings(dim=dim)
    chunks = [
        Chunk(id=f"c{i}", text=text, source="/tmp/note.md", doc_id="doc-1", chunk_index=i)
        for i, text in enumerate(texts)
    ]
    store = VectorStore(tmp_path / name)
    store.add(chunks, embedder.embed_documents(texts))
    return store, embedder, chunks


def test_add_and_search(tmp_path):
    store, embedder, _ = _build_store(
        tmp_path,
        ["TCP 三次握手与四次挥手是计算机网络的重点", "数据结构中的红黑树平衡规则"],
    )
    hits = store.search(embedder.embed_query("计算机网络 TCP 握手"), top_k=2)
    assert hits
    assert "握手" in hits[0].chunk.text
    assert hits[0].score > 0.3


def test_search_on_empty_store(tmp_path):
    store = VectorStore(tmp_path / "empty")
    assert store.is_empty()
    assert store.search(np.ones(8), top_k=3) == []


def test_threshold_filters(tmp_path):
    store, embedder, _ = _build_store(tmp_path, ["完全无关的内容：香蕉苹果橘子"])
    hits = store.search(embedder.embed_query("TCP 握手"), top_k=2, score_threshold=0.9)
    assert hits == []


def test_dimension_mismatch_raises(tmp_path):
    store, _, _ = _build_store(tmp_path, ["第一段文本"])
    with pytest.raises(VectorStoreError):
        store.add(
            [Chunk(id="x", text="第二段", source="/tmp/a.md", doc_id="doc-2", chunk_index=0)],
            np.ones((1, 128), dtype=np.float32),
        )


def test_save_and_load(tmp_path):
    store, embedder, chunks = _build_store(tmp_path, ["保存测试一", "保存测试二"])
    store.save()
    reloaded = VectorStore.load(store.path)
    assert len(reloaded) == 2
    assert reloaded.dim == store.dim
    assert [c.text for c in reloaded.chunks] == [c.text for c in chunks]
    assert np.allclose(reloaded.vectors, store.vectors)


def test_query_dimension_mismatch(tmp_path):
    store, _, _ = _build_store(tmp_path, ["文本"], dim=256)
    with pytest.raises(VectorStoreError):
        store.search(np.ones(16, dtype=np.float32))


def test_neighbors(tmp_path):
    store, _, _ = _build_store(tmp_path, ["块零", "块一", "块二"])
    middle = store.chunks[1]
    neighbors = store.neighbors(middle, radius=1)
    assert [c.chunk_index for c in neighbors] == [0, 2]


def test_remove_document(tmp_path):
    store, _, _ = _build_store(tmp_path, ["a", "b"])
    assert store.remove_document("doc-1") == 2
    assert store.is_empty()
    assert store.vectors.shape[0] == 0
    assert store.remove_document("doc-1") == 0


def test_chunks_for_doc(tmp_path):
    store, embedder, _ = _build_store(tmp_path, ["块零", "块一"])
    store.add(
        [
            Chunk(id="x", text="另一文档", source="/tmp/b.md", doc_id="doc-2", chunk_index=0)
        ],
        embedder.embed_documents(["另一文档"]),
    )
    assert [chunk.text for chunk in store.chunks_for_doc("doc-1")] == ["块零", "块一"]
    assert [chunk.text for chunk in store.chunks_for_doc("doc-2")] == ["另一文档"]
    assert store.chunks_for_doc("missing") == []


def test_collection_manager(tmp_path):
    manager = CollectionManager(tmp_path / "collections")
    assert manager.list_collections() == []

    store = VectorStore(manager.path_of("毕业论文"))
    store.add(
        [Chunk(id="a", text="论文正文", source="/tmp/x.md", doc_id="d", chunk_index=0)],
        np.ones((1, 4), dtype=np.float32),
    )
    store.save()
    manager.ensure("课堂笔记").path.mkdir(parents=True, exist_ok=True)
    (manager.path_of("课堂笔记") / "meta.json").write_text('{"name": "课堂笔记"}', encoding="utf-8")

    names = manager.list_collections()
    assert "毕业论文" in names
    assert "课堂笔记" in names
    assert manager.get("毕业论文").stats()["chunks"] == 1
    assert manager.delete("毕业论文") is True
    assert manager.path_of("毕业论文").exists() is False


def test_collection_name_sanitized(tmp_path):
    manager = CollectionManager(tmp_path / "collections")
    assert manager.normalize_name("a/b") == "a_b"
    with pytest.raises(VectorStoreError):
        manager.normalize_name("   ")
