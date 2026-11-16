"""文本切分：优先在段落/句子边界断开，再按长度合并。

为什么不用第三方 splitter：
- 中英文混排时 LangChain 默认分隔符表对中文标点覆盖不足；
- 这里的分隔符表把中文句末标点放在靠前位置，切出来的块更接近"完整语义单元"；
- 逻辑只有 100 行左右，方便阅读和二次修改。

注意：为保证相邻块有上下文重叠，最终块长度上限是 ``chunk_size + chunk_overlap``。
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

from .models import Chunk, Document, short_hash

DEFAULT_SEPARATORS: Sequence[str] = ("\n\n", "\n", "。", "！", "？", "；", "，", " ", "")


class TextSplitter:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        separators: Optional[Sequence[str]] = None,
        min_chars: int = 30,
        keep_separator: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正整数")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须满足 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = tuple(separators) if separators else DEFAULT_SEPARATORS
        self.min_chars = min_chars
        self.keep_separator = keep_separator

    # ---------------------------------------------------------------- 对外
    def split_text(self, text: str) -> List[str]:
        text = self._normalize(text)
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        pieces = self._split_recursive(text, self.separators)
        merged = self._merge(pieces)
        return self._add_overlap(merged)

    def split_documents(self, documents: Sequence[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for document in documents:
            texts = self.split_text(document.text)
            if len(texts) > 1:
                texts = [t for t in texts if len(t.strip()) >= self.min_chars] or texts
            for index, chunk_text in enumerate(texts):
                chunks.append(
                    Chunk(
                        id=short_hash(f"{document.doc_id}:{document.page}:{index}:{chunk_text[:64]}"),
                        text=chunk_text,
                        source=document.source,
                        doc_id=document.doc_id,
                        chunk_index=index,
                        page=document.page,
                        meta=dict(document.meta),
                    )
                )
        return chunks

    # ---------------------------------------------------------------- 内部
    @staticmethod
    def _normalize(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _split_recursive(self, text: str, separators: Sequence[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[0] if separators else ""
        remaining = separators[1:]

        if not separator:  # 已无分隔符可用：硬切
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(separator)
        if len(parts) == 1:  # 该分隔符不生效，换下一个
            return self._split_recursive(text, remaining)
        if self.keep_separator:
            parts = [parts[i] + separator for i in range(len(parts) - 1)] + [parts[-1]]

        output: List[str] = []
        buffer = ""
        for part in parts:
            if not part:
                continue
            if len(part) > self.chunk_size:
                if buffer:
                    output.append(buffer)
                    buffer = ""
                if remaining:
                    output.extend(self._split_recursive(part, remaining))
                else:
                    output.extend(part[i : i + self.chunk_size] for i in range(0, len(part), self.chunk_size))
            elif len(buffer) + len(part) <= self.chunk_size:
                buffer += part
            else:
                if buffer:
                    output.append(buffer)
                buffer = part
        if buffer:
            output.append(buffer)
        return output

    def _merge(self, pieces: Sequence[str]) -> List[str]:
        chunks: List[str] = []
        buffer = ""
        for piece in pieces:
            if not piece:
                continue
            if len(buffer) + len(piece) <= self.chunk_size:
                buffer += piece
            else:
                if buffer.strip():
                    chunks.append(buffer.strip())
                buffer = piece
        if buffer.strip():
            chunks.append(buffer.strip())
        return chunks

    def _add_overlap(self, chunks: Sequence[str]) -> List[str]:
        if self.chunk_overlap <= 0 or len(chunks) < 2:
            return list(chunks)
        output = [chunks[0]]
        for index in range(1, len(chunks)):
            tail = chunks[index - 1][-self.chunk_overlap :]
            output.append(f"{tail}{chunks[index]}")
        return output


def make_splitter(chunking: dict) -> TextSplitter:
    """从配置 section 构造 splitter，便于 CLI / UI 共用。"""
    separators = chunking.get("separators")
    return TextSplitter(
        chunk_size=int(chunking.get("size", 800)),
        chunk_overlap=int(chunking.get("overlap", 120)),
        separators=tuple(separators) if separators else None,
        min_chars=int(chunking.get("min_chars", 30)),
    )
