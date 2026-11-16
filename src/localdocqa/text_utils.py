"""文本展示工具：清理 Markdown 标记、按查询词高亮命中片段。

全部是纯函数，便于单元测试，也让界面层不必关心 HTML 转义细节。
"""
from __future__ import annotations

import html
import re
from typing import List

MAX_HIGHLIGHT_PHRASES = 6


def clean_text(text: str) -> str:
    """去掉 Markdown 标题符号并折叠空白，用于来源预览。"""
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    return " ".join(text.split())


def preview(text: str, max_chars: int = 220) -> str:
    cleaned = clean_text(text)
    if len(cleaned) > max_chars:
        return cleaned[:max_chars] + "…"
    return cleaned


def _shared_phrases(text: str, query: str, min_len: int = 2, max_len: int = 10) -> List[str]:
    """取查询与原文共有的最长短语，而不是把整句问题丢进去高亮。"""
    query_clean = re.sub(r"\s+", "", query or "")
    query_clean = re.sub(r"[^\w\u4e00-\u9fff]+", "", query_clean)
    if len(query_clean) < min_len:
        return []
    for length in range(min(max_len, len(query_clean)), min_len - 1, -1):
        phrases: List[str] = []
        for start in range(len(query_clean) - length + 1):
            candidate = query_clean[start : start + length]
            if candidate in text and candidate not in phrases:
                phrases.append(candidate)
        if phrases:
            return phrases[:MAX_HIGHLIGHT_PHRASES]
    return []


def highlight(text: str, query: str) -> str:
    """先转义成安全 HTML，再把共有短语包进 <mark>。"""
    escaped = html.escape(text)
    phrases = _shared_phrases(text, query)
    if not phrases:
        return escaped
    escaped_phrases = [html.escape(p) for p in phrases]
    pattern = re.compile("|".join(re.escape(p) for p in sorted(escaped_phrases, key=len, reverse=True)))
    return pattern.sub(lambda match: f"<mark>{match.group(0)}</mark>", escaped)
