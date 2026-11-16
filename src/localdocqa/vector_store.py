"""向量库：numpy 矩阵 + 余弦相似度，落盘为"人类可读 + 可复用"的三个文件。

目录结构（每个集合一个目录）::

    data/collections/<名字>/
        chunks.jsonl      每行一个 chunk（含来源、页码、位置）
        vectors.npy       与 chunks 一一对应的归一化向量
        meta.json         维度、条数、最近更新时间

选择自建而不是 Chroma/FAISS 的理由：
- 依赖极轻（只要 numpy），学生电脑 clone 下来就能跑；
- 数据格式透明，出问题能直接打开文件排查；
- 词条量在 1e4~1e5 级别时，矩阵点积的延迟完全可接受（毫秒级）。
更大规模时可替换为 FAISS，接口保持 `search()` 不变即可。
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .models import Chunk, SearchHit, short_hash

CHUNKS_FILE = "chunks.jsonl"
VECTORS_FILE = "vectors.npy"
META_FILE = "meta.json"


class VectorStoreError(RuntimeError):
    """向量库使用错误（维度不符、数据损坏等）。"""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


class VectorStore:
    def __init__(self, path: str | Path, dim: Optional[int] = None) -> None:
        self.path = Path(path)
        self.dim = int(dim) if dim else None
        self.chunks: List[Chunk] = []
        self.vectors: np.ndarray = np.zeros((0, self.dim or 0), dtype=np.float32)
        self._doc_index: Dict[str, List[int]] = {}

    # ---------------------------------------------------------------- 属性
    @property
    def name(self) -> str:
        return self.path.name

    def __len__(self) -> int:
        return len(self.chunks)

    def is_empty(self) -> bool:
        return not self.chunks

    # ---------------------------------------------------------------- 写入
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        if len(chunks) == 0:
            return 0
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[0] != len(chunks):
            raise VectorStoreError(f"向量条数({vectors.shape[0]}) 与 chunk 条数({len(chunks)}) 不一致")

        new_dim = int(vectors.shape[1])
        if self.dim is None:
            self.dim = new_dim
        elif self.dim != new_dim:
            raise VectorStoreError(
                f"向量维度不一致：库为 {self.dim} 维，新数据为 {new_dim} 维。"
                "更换 embedding 模型后请新建集合或删除旧数据重新入库。"
            )

        if self.vectors.size == 0:
            self.vectors = vectors.astype(np.float32)
        else:
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
        start = len(self.chunks)
        self.chunks.extend(chunks)
        self._rebuild_index(start)
        return len(chunks)

    def _rebuild_index(self, start: int = 0) -> None:
        for position in range(start, len(self.chunks)):
            chunk = self.chunks[position]
            self._doc_index.setdefault(chunk.doc_id, []).append(position)

    # ---------------------------------------------------------------- 检索
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        exclude_ids: Optional[Iterable[str]] = None,
    ) -> List[SearchHit]:
        if self.is_empty():
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if self.dim and query.shape[0] != self.dim:
            raise VectorStoreError(f"查询向量维度 {query.shape[0]} 与库维度 {self.dim} 不一致")
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return []
        query = query / norm

        scores = self.vectors @ query
        order = np.argsort(-scores)
        excluded = set(exclude_ids or [])
        hits: List[SearchHit] = []
        for position in order:
            chunk = self.chunks[int(position)]
            if chunk.id in excluded:
                continue
            score = float(scores[int(position)])
            if score_threshold is not None and score < float(score_threshold):
                break  # 分数已单调下降，后面的都不会更高
            hits.append(SearchHit(chunk=chunk, score=score))
            if len(hits) >= int(top_k):
                break
        return hits

    def chunks_for_doc(self, doc_id: str) -> List[Chunk]:
        """按顺序返回某文档的全部 chunk，供界面查看原文。"""
        return sorted(
            (chunk for chunk in self.chunks if chunk.doc_id == doc_id),
            key=lambda chunk: chunk.chunk_index,
        )

    def neighbors(self, chunk: Chunk, radius: int = 1) -> List[Chunk]:
        """取同一文档中相邻的 chunk，用于补全被切断的上下文。"""
        if radius <= 0:
            return []
        positions = self._doc_index.get(chunk.doc_id, [])
        result: List[Chunk] = []
        for position in positions:
            candidate = self.chunks[position]
            if candidate.chunk_index == chunk.chunk_index:
                continue
            if abs(candidate.chunk_index - chunk.chunk_index) <= radius:
                result.append(candidate)
        return sorted(result, key=lambda c: c.chunk_index)

    # ---------------------------------------------------------------- 统计
    def stats(self) -> Dict[str, object]:
        documents: Dict[str, Dict[str, object]] = {}
        for chunk in self.chunks:
            entry = documents.setdefault(
                chunk.doc_id, {"source": chunk.source, "chunks": 0, "pages": set()}
            )
            entry["chunks"] = int(entry["chunks"]) + 1
            if chunk.page:
                entry["pages"].add(chunk.page)
        documents_list = [
            {
                "doc_id": doc_id,
                "source": info["source"],
                "filename": Path(str(info["source"])).name,
                "chunks": info["chunks"],
                "pages": len(info["pages"]),
            }
            for doc_id, info in documents.items()
        ]
        return {
            "name": self.name,
            "chunks": len(self.chunks),
            "documents": len(documents_list),
            "dim": self.dim or 0,
            "documents_detail": documents_list,
        }

    def remove_document(self, doc_id: str) -> int:
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        removed = len(self.chunks) - len(keep)
        if removed == 0:
            return 0
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if keep else np.zeros((0, self.dim or 0), dtype=np.float32)
        self._doc_index = {}
        self._rebuild_index(0)
        return removed

    # ---------------------------------------------------------------- 持久化
    def save(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        chunks_tmp = self.path / (CHUNKS_FILE + ".tmp")
        with chunks_tmp.open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        chunks_tmp.replace(self.path / CHUNKS_FILE)

        vectors_tmp = self.path / (VECTORS_FILE + ".tmp")
        with vectors_tmp.open("wb") as handle:
            np.save(handle, self.vectors.astype(np.float32))
        vectors_tmp.replace(self.path / VECTORS_FILE)

        meta = {
            "name": self.name,
            "chunks": len(self.chunks),
            "dim": self.dim or 0,
            "documents": len(self._doc_index),
            "updated_at": _now(),
        }
        meta_path = self.path / (META_FILE + ".tmp")
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
        meta_path.replace(self.path / META_FILE)
        return self.path

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        directory = Path(path)
        store = cls(directory)
        meta_path = directory / META_FILE
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                store.dim = int(meta.get("dim") or 0) or None
            except (ValueError, OSError):
                store.dim = None

        chunks_path = directory / CHUNKS_FILE
        if chunks_path.is_file():
            with chunks_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        store.chunks.append(Chunk.from_dict(json.loads(line)))
                    except (ValueError, KeyError) as exc:
                        raise VectorStoreError(f"chunks.jsonl 数据损坏：{exc}") from exc

        vectors_path = directory / VECTORS_FILE
        if vectors_path.is_file():
            # allow_pickle=False：向量必须是纯数值，避免反序列化任意对象
            store.vectors = np.load(vectors_path, allow_pickle=False).astype(np.float32)
            if store.vectors.ndim == 1:
                store.vectors = store.vectors.reshape(-1, 1)
            if store.dim is None and store.vectors.size:
                store.dim = int(store.vectors.shape[1])

        if len(store.chunks) and len(store.vectors) and len(store.chunks) != len(store.vectors):
            raise VectorStoreError(
                f"向量库数据不一致：{len(store.chunks)} 条 chunk，{len(store.vectors)} 条向量。"
                "请删除该集合后重新入库。"
            )
        if not store.chunks:
            store.vectors = np.zeros((0, store.dim or 0), dtype=np.float32)
        store._rebuild_index(0)
        return store


class CollectionManager:
    """集合（知识库）管理器：一个集合 = 一个可独立问答的文档空间。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize_name(name: str) -> str:
        cleaned = str(name or "").strip().replace(" ", "_")
        if not cleaned:
            raise VectorStoreError("集合名称不能为空")
        for bad in ("/", "\\", "..", ":", "*", "?", '"', "<", ">", "|"):
            cleaned = cleaned.replace(bad, "_")
        if len(cleaned) > 60:
            cleaned = cleaned[:60]
        return cleaned

    def list_collections(self) -> List[str]:
        if not self.root.exists():
            return []
        result = []
        for child in sorted(self.root.iterdir()):
            if child.is_dir() and (child / META_FILE).is_file():
                result.append(child.name)
        return result

    def exists(self, name: str) -> bool:
        return (self.root / self.normalize_name(name)).is_dir()

    def path_of(self, name: str) -> Path:
        return self.root / self.normalize_name(name)

    def get(self, name: str) -> VectorStore:
        return VectorStore.load(self.path_of(name))

    def ensure(self, name: str) -> VectorStore:
        """取集合，不存在则创建空目录（首次入库时落盘）。"""
        store = self.get(name)
        store.path.mkdir(parents=True, exist_ok=True)
        return store

    def delete(self, name: str) -> bool:
        target = self.path_of(name)
        if not target.is_dir():
            return False
        shutil.rmtree(target)
        return True

    def stats(self) -> List[Dict[str, object]]:
        result = []
        for name in self.list_collections():
            try:
                result.append(self.get(name).stats())
            except VectorStoreError as exc:  # 损坏的集合不应导致整体崩溃
                result.append({"name": name, "error": str(exc), "chunks": 0, "documents": 0})
        return result

    def new_collection_id(self) -> str:
        return short_hash(_now() + str(self.list_collections()), 6)
