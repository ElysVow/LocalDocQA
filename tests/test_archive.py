from __future__ import annotations

import io
import zipfile

import pytest

from localdocqa.archive import export_collection, import_collection
from localdocqa.config import Config
from localdocqa.embeddings import HashingEmbeddings
from localdocqa.ingest import ingest
from localdocqa.vector_store import CollectionManager, VectorStoreError


@pytest.fixture()
def config(tmp_path):
    cfg = Config({}, root=tmp_path)
    cfg.set("chunking.size", 160)
    cfg.set("chunking.overlap", 20)
    return cfg


@pytest.fixture()
def docs_dir(tmp_path):
    directory = tmp_path / "docs"
    directory.mkdir()
    (directory / "a.md").write_text(
        "# 实验规范\n\n每次实验需要提交实验报告、源码压缩包与运行说明三部分。" * 4,
        encoding="utf-8",
    )
    return directory


def _ingest_demo(config, docs_dir):
    return ingest([str(docs_dir)], collection="demo", config=config, embeddings=HashingEmbeddings())


def test_export_contains_expected_files(config, docs_dir):
    _ingest_demo(config, docs_dir)
    payload = export_collection("demo", config)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = set(archive.namelist())
    assert {"chunks.jsonl", "vectors.npy", "meta.json", "documents.json", "collection.json"} <= names


def test_export_empty_collection_raises(config):
    with pytest.raises(VectorStoreError):
        export_collection("demo", config)


def test_import_roundtrip_keeps_chunks(config, docs_dir):
    _ingest_demo(config, docs_dir)
    original = CollectionManager(config.collections_dir()).get("demo")
    payload = export_collection("demo", config)

    imported = import_collection(payload, config, name="restored")
    restored = CollectionManager(config.collections_dir()).get(imported)
    assert len(restored) == len(original)
    assert restored.dim == original.dim
    assert (config.collections_dir() / imported / "documents.json").is_file()


def test_import_name_conflict_gets_suffix(config, docs_dir):
    _ingest_demo(config, docs_dir)
    payload = export_collection("demo", config)
    first = import_collection(payload, config, name="restored")
    second = import_collection(payload, config, name="restored")
    assert first == "restored"
    assert second == "restored-2"


def test_import_rejects_non_zip(config):
    with pytest.raises(VectorStoreError):
        import_collection(b"this is not a zip", config)


def test_import_rejects_zip_without_data(config):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "nothing useful")
    with pytest.raises(VectorStoreError):
        import_collection(buffer.getvalue(), config)
