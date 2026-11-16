"""文档入库：解析 → 去重 → 切分 → 向量化 → 落盘 → 更新文档登记表。

去重口径是"文件内容 SHA256"：同一个文件改个名字再导入不会重复占用向量库；
文件内容变了（哈希变了）才会作为新文档入库，旧版本需要手动删除。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .config import Config
from .embeddings import EmbeddingError, EmbeddingProvider, build_embeddings
from .loader import file_sha256, load_file, resolve_inputs
from .logging_utils import get_logger
from .splitter import make_splitter
from .vector_store import CollectionManager, VectorStore, VectorStoreError

logger = get_logger("localdocqa.ingest")

REGISTRY_FILE = "documents.json"
ProgressFn = Optional[Callable[[str, Optional[float]], None]]


@dataclass
class IngestReport:
    collection: str
    files: int = 0
    new_files: int = 0
    skipped_files: int = 0
    chunks: int = 0
    elapsed: float = 0.0
    errors: List[Dict[str, str]] = field(default_factory=list)
    details: List[Dict[str, object]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def summary(self) -> str:
        parts = [
            f"集合 {self.collection}",
            f"扫描 {self.files} 个文件",
            f"新增 {self.new_files}",
            f"跳过重复 {self.skipped_files}",
            f"写入 {self.chunks} 个 chunk",
            f"耗时 {self.elapsed:.1f}s",
        ]
        if self.errors:
            parts.append(f"失败 {len(self.errors)}")
        return "，".join(parts)


# ------------------------------------------------------------------ 登记表
def registry_path(store: VectorStore) -> Path:
    return store.path / REGISTRY_FILE


def load_registry(store: VectorStore) -> List[Dict[str, object]]:
    path = registry_path(store)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        logger.warning("文档登记表损坏，按空表处理：%s", exc)
        return []
    documents = payload.get("documents") if isinstance(payload, dict) else None
    return list(documents or [])


def save_registry(store: VectorStore, documents: Sequence[Dict[str, object]]) -> None:
    path = registry_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "collection": store.name,
        "documents": list(documents),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------------ 入库
def ingest(
    paths: Sequence[str | Path],
    collection: str = "default",
    config: Optional[Config] = None,
    force: bool = False,
    embeddings: Optional[EmbeddingProvider] = None,
    progress: ProgressFn = None,
) -> IngestReport:
    """把文件/目录导入指定集合，返回本次入库统计。"""
    config = config or Config.load()
    report = IngestReport(collection=CollectionManager.normalize_name(collection))

    def notify(message: str, fraction: Optional[float] = None) -> None:
        logger.info(message)
        if progress:
            progress(message, fraction)

    started = time.perf_counter()
    manager = CollectionManager(config.collections_dir())
    store = manager.ensure(report.collection)
    registry = load_registry(store)

    by_sha: Dict[str, Dict[str, object]] = {str(d.get("sha256")): d for d in registry if d.get("sha256")}
    files = resolve_inputs(paths)
    report.files = len(files)
    if not files:
        report.elapsed = time.perf_counter() - started
        notify("没有找到可处理的文件（支持 pdf/docx/md/txt/csv/json/html）")
        return report

    try:
        embedder = embeddings or build_embeddings(config.section("embeddings"))
    except EmbeddingError as exc:
        report.errors.append({"file": "<embedding>", "error": str(exc)})
        report.elapsed = time.perf_counter() - started
        return report

    splitter = make_splitter(config.section("chunking"))
    notify(f"开始处理 {len(files)} 个文件，embedding={embedder.name}")

    for position, file_path in enumerate(files, start=1):
        label = f"[{position}/{len(files)}] {file_path.name}"
        notify(label, (position - 1) / len(files))
        try:
            sha = file_sha256(file_path)
            existing = by_sha.get(sha)
            if existing and not force:
                report.skipped_files += 1
                report.details.append(
                    {"file": file_path.name, "status": "跳过", "chunks": 0, "reason": "内容与已入库文件相同"}
                )
                logger.info("跳过重复文件：%s（已入库为 %s）", file_path.name, existing.get("doc_id"))
                continue
            if existing and force:
                removed = store.remove_document(str(existing.get("doc_id")))
                registry = [d for d in registry if str(d.get("doc_id")) != str(existing.get("doc_id"))]
                by_sha.pop(sha, None)
                logger.info("强制重建 %s，移除旧 chunk %s 个", file_path.name, removed)

            documents = load_file(file_path)
            chunks = splitter.split_documents(documents)
            if not chunks:
                raise ValueError("切分后没有得到有效文本")
            vectors = embedder.embed_documents([c.text for c in chunks])
            store.add(chunks, vectors)

            entry = {
                "doc_id": documents[0].doc_id,
                "source": str(file_path),
                "filename": file_path.name,
                "sha256": sha,
                "size": file_path.stat().st_size,
                "chunks": len(chunks),
                "pages": sum(1 for d in documents if d.page),
                "ingested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "embedding": embedder.name,
            }
            registry.append(entry)
            by_sha[sha] = entry
            report.new_files += 1
            report.chunks += len(chunks)
            report.details.append(
                {"file": file_path.name, "status": "新增", "chunks": len(chunks), "reason": ""}
            )
        except (OSError, ValueError, RuntimeError, VectorStoreError) as exc:
            logger.warning("处理失败：%s —— %s", file_path.name, exc)
            report.errors.append({"file": str(file_path), "error": str(exc)})
            report.details.append(
                {"file": file_path.name, "status": "失败", "chunks": 0, "reason": str(exc)}
            )

    if report.new_files:
        store.save()
        save_registry(store, registry)
    report.elapsed = time.perf_counter() - started
    notify(
        f"完成：新增 {report.new_files} 个文件 / {report.chunks} 个 chunk"
        + (f"，失败 {len(report.errors)} 个" if report.errors else ""),
        1.0,
    )
    return report


def remove_document(collection: str, doc_id: str, config: Optional[Config] = None) -> int:
    """删除某文档的全部 chunk，返回移除的 chunk 数量。"""
    config = config or Config.load()
    manager = CollectionManager(config.collections_dir())
    store = manager.get(collection)
    removed = store.remove_document(doc_id)
    if removed:
        registry = [d for d in load_registry(store) if str(d.get("doc_id")) != doc_id]
        save_registry(store, registry)
        store.save()
    return removed


def list_documents(collection: str, config: Optional[Config] = None) -> List[Dict[str, object]]:
    config = config or Config.load()
    manager = CollectionManager(config.collections_dir())
    store = manager.get(collection)
    registry = load_registry(store)
    if registry:
        return registry
    # 老数据没有登记表时，用向量库里的 chunk 反推，保证界面不至于空白
    detail = store.stats().get("documents_detail", [])
    return [
        {
            "doc_id": item["doc_id"],
            "source": item["source"],
            "filename": item["filename"],
            "sha256": "",
            "size": 0,
            "chunks": item["chunks"],
            "pages": item["pages"],
            "ingested_at": "",
            "embedding": "",
        }
        for item in detail
    ]
