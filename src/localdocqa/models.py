"""领域模型：Document / Chunk / SearchHit / Answer。

这些对象是各模块之间唯一的数据契约，序列化格式与落盘格式保持一致，
保证"向量库里的数据"和"程序里的对象"不会各说各话。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def short_hash(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


@dataclass
class Document:
    """一段已加载的文本（通常对应一个文件的一页或整个文件）。"""

    text: str
    source: str
    doc_id: str = ""
    page: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = short_hash(f"{self.source}#{self.page}", 12)

    @property
    def filename(self) -> str:
        return Path(self.source).name

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    """切分后的检索单元，是向量库里最小的存储单位。"""

    id: str
    text: str
    source: str
    doc_id: str
    chunk_index: int
    page: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = short_hash(f"{self.doc_id}:{self.chunk_index}:{self.text[:64]}")

    @property
    def filename(self) -> str:
        return Path(self.source).name

    def citation(self) -> str:
        base = self.filename
        if self.page:
            return f"{base} · 第 {self.page} 页"
        return base

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            id=data.get("id", ""),
            text=data["text"],
            source=data["source"],
            doc_id=data["doc_id"],
            chunk_index=int(data["chunk_index"]),
            page=data.get("page"),
            meta=data.get("meta") or {},
        )


@dataclass
class SearchHit:
    """一次检索命中的结果。"""

    chunk: Chunk
    score: float

    def to_source(self, index: int, preview_chars: int = 220) -> Dict[str, Any]:
        # 预览里去掉 Markdown 标题符号，避免界面上一堆 "##"
        raw = re.sub(r"(?m)^#{1,6}\s*", "", self.chunk.text)
        preview = " ".join(raw.split())
        if len(preview) > preview_chars:
            preview = preview[:preview_chars] + "…"
        return {
            "index": index,
            "source": self.chunk.source,
            "filename": self.chunk.filename,
            "page": self.chunk.page,
            "chunk_index": self.chunk.chunk_index,
            "score": round(float(self.score), 4),
            "preview": preview,
            "citation": self.chunk.citation(),
        }


@dataclass
class Answer:
    """一次问答的完整结果，可直接写入日志或导出 Markdown。"""

    question: str
    text: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    hits: List[SearchHit] = field(default_factory=list)
    elapsed: float = 0.0
    used_llm: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "text": self.text,
            "sources": self.sources,
            "elapsed": round(self.elapsed, 3),
            "used_llm": self.used_llm,
        }
