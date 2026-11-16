"""配置加载：内置默认值 + 用户 config.yaml 深合并 + 环境变量覆盖。

设计原则
--------
1. 任何配置项都有默认值，缺少 config.yaml 也能跑。
2. 相对路径一律相对仓库根目录解析，避免"从哪个目录启动"影响结果。
3. 环境变量只覆盖少量关键项（服务地址、数据目录），便于容器化 / CI。
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# 仓库根目录：src/localdocqa/config.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULTS: Dict[str, Any] = {
    "app": {
        "title": "LocalDocQA 本地文档问答助手",
        "data_dir": "data",
        "log_level": "INFO",
        "history_turns": 3,
    },
    "llm": {
        "provider": "ollama",  # ollama | openai | none
        "base_url": "http://127.0.0.1:11434",
        "model": "qwen2.5:7b",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.2,
        "num_ctx": 4096,
        "timeout": 180,
        "max_context_chars": 6000,
    },
    "embeddings": {
        "provider": "ollama",  # ollama | hash
        "base_url": "http://127.0.0.1:11434",
        "model": "nomic-embed-text",
        "batch_size": 16,
        "dim": 256,
        "fallback_to_hash": True,
    },
    "chunking": {
        "size": 800,
        "overlap": 120,
        "min_chars": 30,
        "separators": ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    },
    "retrieval": {
        "top_k": 5,
        "score_threshold": 0.25,
        "lexical_relative_cutoff": 0.25,  # 离线词法向量模式下的相对阈值（相对最高分）
        "expand_neighbors": 1,
        "max_chunks_in_context": 8,
    },
    "storage": {
        "dir": "data",
    },
}

DEFAULT_CONFIG_FILES = ("config.yaml", "config.example.yaml")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并字典，override 优先；不修改入参。"""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    """轻量配置对象，按 section 访问。"""

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        root: Optional[Path] = None,
        source: Optional[Path] = None,
    ) -> None:
        self._data = _deep_merge(DEFAULTS, data or {})
        self.root = Path(root or REPO_ROOT).resolve()
        self.source = Path(source).resolve() if source else None
        self._apply_env()

    # ------------------------------------------------------------------ 加载
    @classmethod
    def load(
        cls,
        path: Optional[str | Path] = None,
        root: Optional[str | Path] = None,
    ) -> "Config":
        base = Path(root).resolve() if root else REPO_ROOT
        candidates: List[Path] = []
        explicit = path or os.environ.get("LOCALDOCQA_CONFIG")
        if explicit:
            candidates = [Path(explicit)]
        else:
            candidates = [base / name for name in DEFAULT_CONFIG_FILES]

        data: Dict[str, Any] = {}
        used: Optional[Path] = None
        for candidate in candidates:
            if candidate.is_file():
                with candidate.open("r", encoding="utf-8") as handle:
                    loaded = yaml.safe_load(handle) or {}
                if not isinstance(loaded, dict):
                    raise ValueError(f"配置文件格式错误，顶层应为映射: {candidate}")
                data = loaded
                used = candidate
                break
        return cls(data, root=base, source=used)

    def _apply_env(self) -> None:
        ollama_url = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST")
        if ollama_url:
            if not ollama_url.startswith(("http://", "https://")):
                ollama_url = f"http://{ollama_url}"
            self.set("llm.base_url", ollama_url.rstrip("/"))
            self.set("embeddings.base_url", ollama_url.rstrip("/"))
        data_dir = os.environ.get("LOCALDOCQA_DATA_DIR")
        if data_dir:
            self.set("app.data_dir", data_dir)
            self.set("storage.dir", data_dir)

    # ------------------------------------------------------------------ 访问
    def section(self, name: str) -> Dict[str, Any]:
        return copy.deepcopy(self._data.get(name, {}))

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    def save(self, path: Optional[str | Path] = None) -> Path:
        target = Path(path) if path else (self.source or self.root / "config.yaml")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self._data, handle, allow_unicode=True, sort_keys=False)
        return target

    # ------------------------------------------------------------------ 路径
    def data_dir(self) -> Path:
        raw = Path(self.get("app.data_dir", "data"))
        resolved = raw if raw.is_absolute() else (self.root / raw)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved.resolve()

    def collections_dir(self) -> Path:
        path = Path(self.get("storage.dir", "data"))
        if not path.is_absolute():
            path = self.root / path
        path = path / "collections"
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        src = self.source.name if self.source else "<defaults>"
        return f"Config(root={self.root}, source={src})"
