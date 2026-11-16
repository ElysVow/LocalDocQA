from __future__ import annotations

import json

import pytest

from localdocqa.config import Config
from localdocqa.embeddings import HashingEmbeddings
from localdocqa.ingest import ingest, list_documents, remove_document


@pytest.fixture()
def config(tmp_path):
    cfg = Config({}, root=tmp_path)
    cfg.set("chunking.size", 200)
    cfg.set("chunking.overlap", 20)
    cfg.set("retrieval.score_threshold", -1.0)
    return cfg


@pytest.fixture()
def docs_dir(tmp_path):
    directory = tmp_path / "docs"
    directory.mkdir()
    (directory / "a.md").write_text(
        "# 项目说明\n\n" + "本项目的验收标准包括三条：功能可用、文档齐全、测试通过。" * 6,
        encoding="utf-8",
    )
    (directory / "b.txt").write_text(
        "课程笔记：计算机网络的 TCP 三次握手流程包含 SYN、SYN+ACK、ACK 三个报文。",
        encoding="utf-8",
    )
    return directory


def test_ingest_creates_collection(config, docs_dir):
    report = ingest([str(docs_dir)], collection="demo", config=config, embeddings=HashingEmbeddings())
    assert report.new_files == 2
    assert report.chunks > 0
    assert not report.errors
    assert (config.collections_dir() / "demo" / "vectors.npy").is_file()
    assert report.summary().startswith("集合 demo")


def test_ingest_skips_duplicates(config, docs_dir):
    embeddings = HashingEmbeddings()
    ingest([str(docs_dir)], collection="demo", config=config, embeddings=embeddings)
    report = ingest([str(docs_dir)], collection="demo", config=config, embeddings=embeddings)
    assert report.new_files == 0
    assert report.skipped_files == 2


def test_ingest_force_rebuild(config, docs_dir):
    embeddings = HashingEmbeddings()
    ingest([str(docs_dir)], collection="demo", config=config, embeddings=embeddings)
    first_chunks = len(list_documents("demo", config))
    report = ingest([str(docs_dir)], collection="demo", config=config, embeddings=embeddings, force=True)
    assert report.new_files == 2
    # 强制重建后文档数量不翻倍
    assert len(list_documents("demo", config)) == first_chunks


def test_ingest_records_errors(config, tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{不是合法 JSON", encoding="utf-8")
    report = ingest([str(broken)], collection="demo", config=config, embeddings=HashingEmbeddings())
    assert report.errors
    assert "JSON 解析失败" in report.errors[0]["error"]


def test_ingest_empty_paths(config):
    report = ingest(["/path/that/does/not/exist"], collection="demo", config=config)
    assert report.files == 0
    assert report.new_files == 0


def test_list_and_remove_document(config, docs_dir):
    ingest([str(docs_dir)], collection="demo", config=config, embeddings=HashingEmbeddings())
    documents = list_documents("demo", config)
    assert len(documents) == 2
    assert {d["filename"] for d in documents} == {"a.md", "b.txt"}
    removed = remove_document("demo", str(documents[0]["doc_id"]), config)
    assert removed > 0
    assert len(list_documents("demo", config)) == 1


def test_registry_file_is_valid_json(config, docs_dir):
    ingest([str(docs_dir)], collection="demo", config=config, embeddings=HashingEmbeddings())
    path = config.collections_dir() / "demo" / "documents.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["collection"] == "demo"
    assert len(payload["documents"]) == 2
