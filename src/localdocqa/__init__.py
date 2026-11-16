"""LocalDocQA：完全本地运行的文档问答（RAG）工具。

对外暴露的核心能力：
    load_file / resolve_inputs   —— 文件加载
    TextSplitter                —— 文本切分
    CollectionManager           —— 向量库管理
    RAGEngine                   —— 检索 + 生成
"""
from __future__ import annotations

__version__ = "0.1.0"

from .config import Config  # noqa: F401
from .models import Answer, Chunk, Document, SearchHit  # noqa: F401

__all__ = ["Config", "Document", "Chunk", "SearchHit", "Answer", "__version__"]
