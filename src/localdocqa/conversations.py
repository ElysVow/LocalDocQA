"""对话记录落盘：刷新页面或重启服务后，仍能接着上次的对话继续问。

按集合分开存，路径 `data/conversations/<集合>.json`；写入走临时文件再替换，
避免进程被杀时留下半个 JSON。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config
from .logging_utils import get_logger
from .vector_store import CollectionManager

logger = get_logger("localdocqa.conversations")

MAX_MESSAGES = 200
VALID_ROLES = {"user", "assistant"}


def conversations_dir(config: Optional[Config] = None) -> Path:
    config = config or Config.load()
    return config.data_dir() / "conversations"


def conversation_path(collection: str, config: Optional[Config] = None) -> Path:
    name = CollectionManager.normalize_name(collection)
    return conversations_dir(config) / f"{name}.json"


def _normalize(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in VALID_ROLES or not isinstance(content, str):
            continue
        entry: Dict[str, Any] = {"role": role, "content": content}
        sources = message.get("sources")
        if isinstance(sources, list) and sources:
            entry["sources"] = sources
        query = message.get("query")
        if isinstance(query, str) and query:
            entry["query"] = query
        cleaned.append(entry)
    return cleaned[-MAX_MESSAGES:]


def load_conversation(collection: str, config: Optional[Config] = None) -> List[Dict[str, Any]]:
    path = conversation_path(collection, config)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("对话记录损坏，按空记录处理：%s", exc)
        return []
    messages = payload.get("messages") if isinstance(payload, dict) else payload
    if not isinstance(messages, list):
        return []
    return _normalize(messages)


def save_conversation(
    collection: str, messages: List[Dict[str, Any]], config: Optional[Config] = None
) -> Path:
    path = conversation_path(collection, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "collection": CollectionManager.normalize_name(collection),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": _normalize(messages),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def clear_conversation(collection: str, config: Optional[Config] = None) -> bool:
    path = conversation_path(collection, config)
    if not path.is_file():
        return False
    path.unlink()
    return True
