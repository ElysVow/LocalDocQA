"""集合导出 / 导入：把一个集合打包成 zip，换台电脑也能接着用。

只接受白名单文件，且限制解压后体积；导入时先写临时目录并做一次完整加载校验，
校验通过才改名生效，避免把损坏的向量库放进 data/。
"""
from __future__ import annotations

import io
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import Config
from .vector_store import CollectionManager, VectorStore, VectorStoreError

MANIFEST = "collection.json"
DATA_FILES = ("chunks.jsonl", "vectors.npy", "meta.json", "documents.json")
MAX_UNCOMPRESSED = 300 * 1024 * 1024
MAX_NAME_SUFFIX = 50


def export_collection(collection: str, config: Optional[Config] = None) -> bytes:
    config = config or Config.load()
    manager = CollectionManager(config.collections_dir())
    store = manager.get(collection)
    if not (store.path / "chunks.jsonl").is_file() or not (store.path / "vectors.npy").is_file():
        raise VectorStoreError(f"集合「{collection}」还没有可导出的索引，先导入文档再试")
    buffer = io.BytesIO()
    manifest: Dict[str, Any] = {
        "collection": store.name,
        "chunks": len(store.chunks),
        "dim": store.dim or 0,
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))
        for filename in DATA_FILES:
            path = store.path / filename
            if path.is_file():
                archive.writestr(filename, path.read_bytes())
    return buffer.getvalue()


def _unique_name(manager: CollectionManager, desired: str) -> str:
    base = CollectionManager.normalize_name(desired)
    if not manager.path_of(base).exists():
        return base
    for index in range(2, MAX_NAME_SUFFIX + 1):
        candidate = CollectionManager.normalize_name(f"{base}-{index}")
        if not manager.path_of(candidate).exists():
            return candidate
    raise VectorStoreError("同名集合太多，请换一个集合名称再导入")


def import_collection(
    data: bytes, config: Optional[Config] = None, name: Optional[str] = None
) -> str:
    config = config or Config.load()
    manager = CollectionManager(config.collections_dir())
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise VectorStoreError(f"不是有效的 zip 文件：{exc}") from exc

    with archive:
        entries = [info for info in archive.infolist() if not info.is_dir()]
        total = sum(info.file_size for info in entries)
        if total > MAX_UNCOMPRESSED:
            raise VectorStoreError("压缩包解压后超过 300 MB，拒绝导入")

        files: Dict[str, zipfile.ZipInfo] = {}
        manifest: Dict[str, Any] = {}
        for info in entries:
            basename = Path(info.filename).name
            if basename in DATA_FILES and basename not in files:
                files[basename] = info
            elif basename == MANIFEST and not manifest:
                try:
                    manifest = json.loads(archive.read(info).decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    manifest = {}

        if "chunks.jsonl" not in files or "vectors.npy" not in files:
            raise VectorStoreError("压缩包里缺少 chunks.jsonl 或 vectors.npy，无法导入")

        desired = name or str(manifest.get("collection") or "")
        if not desired:
            desired = "imported-" + datetime.now().strftime("%Y%m%d-%H%M")
        target_name = _unique_name(manager, desired)
        target = manager.path_of(target_name)
        staging = target.with_name(target.name + ".importing")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        try:
            for basename, info in files.items():
                (staging / basename).write_bytes(archive.read(info))
            VectorStore.load(staging)  # 完整性校验：条数、维度、JSON 结构
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        staging.rename(target)
    return target_name
