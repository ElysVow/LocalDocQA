from __future__ import annotations

import json

from localdocqa.config import Config
from localdocqa.conversations import (
    MAX_MESSAGES,
    clear_conversation,
    conversation_path,
    load_conversation,
    save_conversation,
)


def _config(tmp_path) -> Config:
    return Config({}, root=tmp_path)


def test_save_and_load_roundtrip(tmp_path):
    config = _config(tmp_path)
    messages = [
        {"role": "user", "content": "实验报告要交什么？"},
        {
            "role": "assistant",
            "content": "三部分：报告、源码、说明。",
            "sources": [{"index": 1, "filename": "03.md"}],
            "query": "实验报告要交什么？",
        },
    ]
    path = save_conversation("demo", messages, config)
    assert path.is_file()
    assert load_conversation("demo", config) == messages


def test_invalid_messages_are_filtered(tmp_path):
    config = _config(tmp_path)
    path = conversation_path("demo", config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "collection": "demo",
        "messages": [
            {"role": "user", "content": "正常问题"},
            {"role": "system", "content": "不该保留"},
            {"role": "assistant", "content": 123},
            "不是字典",
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert load_conversation("demo", config) == [{"role": "user", "content": "正常问题"}]


def test_messages_are_trimmed(tmp_path):
    config = _config(tmp_path)
    messages = [{"role": "user", "content": f"问题 {index}"} for index in range(250)]
    save_conversation("demo", messages, config)
    loaded = load_conversation("demo", config)
    assert len(loaded) == MAX_MESSAGES
    assert loaded[0]["content"] == "问题 50"


def test_clear_conversation(tmp_path):
    config = _config(tmp_path)
    save_conversation("demo", [{"role": "user", "content": "问"}], config)
    assert clear_conversation("demo", config) is True
    assert load_conversation("demo", config) == []
    assert clear_conversation("demo", config) is False


def test_missing_file_returns_empty(tmp_path):
    assert load_conversation("no-such", _config(tmp_path)) == []
