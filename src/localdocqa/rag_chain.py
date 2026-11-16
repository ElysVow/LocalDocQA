"""RAG 主链路：检索 → 上下文拼装 → 生成（支持流式）→ 带引用的回答。

对外只有两个入口：
    engine.answer(question)   —— 返回完整 Answer
    engine.stream(question)   —— 逐段产出事件（供 UI 打字机效果）

所有"没有命中文档 / 没有模型 / 模型报错"的情况都返回可读的说明，不会抛异常中断界面。
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .config import Config
from .embeddings import EmbeddingError, EmbeddingProvider, HashingEmbeddings, build_embeddings
from .llm import LLMClient, LLMError, build_llm
from .logging_utils import get_logger
from .models import Answer, SearchHit
from .vector_store import CollectionManager, VectorStore, VectorStoreError

logger = get_logger("localdocqa.rag")

SYSTEM_PROMPT = """你是 LocalDocQA 的文档问答助手，服务于"只依据用户自己的文档回答问题"这一场景。

必须遵守的规则：
1. 只使用【参考资料】中的信息回答，不得凭记忆或常识补充资料之外的事实。
2. 每个关键结论后面用 [编号] 标注来源，例如 [1]、[2]；编号必须与参考资料一致。
3. 如果资料不足或没有相关内容，直接说明"资料中没有相关内容"，并指出还需要补充什么资料。
4. 用中文作答，先给结论再给依据，条理清晰，避免空话套话。
5. 保留原文中的数字、公式、专有名词，不要改写它们。"""

NO_DOCUMENT_HINT = (
    "这个知识库还没有任何文档。请先在左侧上传 PDF / Word / Markdown / TXT 等文件，"
    "或点击「导入示例文档」，然后再提问。"
)

NO_HIT_HINT = (
    "没有在文档中检索到与问题相关的内容。可以试试：\n"
    "1. 换一种说法，或补充更具体的关键词；\n"
    "2. 在侧边栏调低「相关度阈值」、调大「召回条数」；\n"
    "3. 确认相关文档已经入库。"
)

_SENTENCE_SPLIT = re.compile(r"[。！？；\n]|(?<=[.!?])\s+")
_HEADING_PREFIX = re.compile(r"^#+\s*")


def _usable_sentences(text: str) -> list:
    """把块内文本切成可用的句子：去掉 Markdown 标题符号，过滤过短行与表格/代码行。"""
    result = []
    for raw in _SENTENCE_SPLIT.split(text):
        sentence = _HEADING_PREFIX.sub("", raw.strip()).strip()
        if not sentence or len(sentence) < 12:
            continue
        if sentence.startswith(("|", "```", ">", "- ")) or sentence.count("|") >= 2:
            continue
        result.append(sentence)
    return result


def format_context(hits: Sequence[SearchHit], max_chars: int) -> tuple[str, List[Dict[str, Any]]]:
    """把命中结果拼成带编号的参考资料，同时返回与编号一致的来源列表。"""
    blocks: List[str] = []
    sources: List[Dict[str, Any]] = []
    used = 0
    for index, hit in enumerate(hits, start=1):
        text = hit.chunk.text.strip()
        block = f"[{index}] 来源：{hit.chunk.citation()}\n{text}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        sources.append(hit.to_source(index))
        used += len(block)
    return "\n\n".join(blocks), sources


def _extractive_answer(question: str, hits: Sequence[SearchHit], max_points: int = 3) -> str:
    """无模型时的兜底：从命中文档里挑出与问题最相关的句子原样呈现。

    编号沿用 hits 的原始序号（与 format_context 生成的来源编号一致），
    跳过与问题毫无词重合的段落，避免把凑数的无关文档放进答案。
    """
    question_tokens = set(HashingEmbeddings._tokenize(question))
    picked: list = []
    for index, hit in enumerate(hits, start=1):
        sentences = _usable_sentences(hit.chunk.text) or [hit.chunk.text.strip()[:200]]
        scored = [
            (len(question_tokens & set(HashingEmbeddings._tokenize(sentence))), len(sentence), sentence)
            for sentence in sentences
        ]
        overlap, _, best = max(scored, key=lambda item: (item[0], item[1]))
        if overlap <= 0 and picked:
            continue
        if len(best) > 260:
            best = best[:260] + "…"
        picked.append(f"{len(picked) + 1}. 根据 [{index}]：{best}")
        if len(picked) >= max_points:
            break

    lines = [
        "当前为**离线提取模式**（未连接大模型），以下内容直接从命中文档中摘录，未做归纳："
    ]
    lines.extend(picked)
    return "\n".join(lines)


def _llm_failure_note(config: Config, exc: Exception) -> str:
    """把底层异常翻译成用户能照做的一句话，完整异常只写日志。"""
    base_url = config.get("llm.base_url", "http://127.0.0.1:11434")
    logger.debug("LLM 调用失败 %s: %s", base_url, exc)
    return "\n\n> 本地模型未连接，已改用离线摘录。执行 `ollama serve` 后重试即可获得归纳式回答。"


class RAGEngine:
    """一个 engine 绑定一个集合；可复用于 CLI、Streamlit、脚本和测试。"""

    def __init__(
        self,
        config: Config,
        collection: str = "default",
        embeddings: Optional[EmbeddingProvider] = None,
        llm: Optional[LLMClient] = None,
    ) -> None:
        self.config = config
        self.manager = CollectionManager(config.collections_dir())
        self.collection = CollectionManager.normalize_name(collection)
        self._embeddings = embeddings
        self._llm = llm
        self._llm_ready: Optional[bool] = None if llm is not None else None
        self._store: Optional[VectorStore] = None

    # ------------------------------------------------------------ 组件初始化
    @property
    def embeddings(self) -> EmbeddingProvider:
        if self._embeddings is None:
            self._embeddings = build_embeddings(self.config.section("embeddings"))
        return self._embeddings

    @property
    def llm(self) -> Optional[LLMClient]:
        if self._llm is None and self._llm_ready is None:
            self._llm = build_llm(self.config.section("llm"))
            self._llm_ready = True
        return self._llm

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = self.manager.get(self.collection)
        return self._store

    def reload(self) -> None:
        """外部（如 CLI 入库）改动数据后，刷新内存中的集合。"""
        self._store = None

    def set_collection(self, collection: str) -> None:
        self.collection = CollectionManager.normalize_name(collection)
        self._store = None

    # ------------------------------------------------------------ 检索
    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[SearchHit]:
        question = (question or "").strip()
        if not question or self.store.is_empty():
            return []

        retrieval_cfg = self.config.section("retrieval")
        top_k = int(top_k or retrieval_cfg.get("top_k", 5))
        threshold = retrieval_cfg.get("score_threshold")
        expand = int(retrieval_cfg.get("expand_neighbors", 0))
        max_chunks = int(retrieval_cfg.get("max_chunks_in_context", 8))

        # 离线词法向量的余弦值整体比语义向量低一个量级，用绝对阈值会把几乎所有结果
        # 都过滤掉；这里改用"相对最高分"的截断，只淘汰明显无关的尾部结果。
        lexical = getattr(self.embeddings, "kind", "semantic") == "lexical"
        query_vector = self.embeddings.embed_query(question)
        hits = self.store.search(
            query_vector, top_k=top_k, score_threshold=None if lexical else threshold
        )

        if lexical and hits:
            cutoff = hits[0].score * float(retrieval_cfg.get("lexical_relative_cutoff", 0.4))
            filtered = [hit for hit in hits if hit.score >= cutoff]
            hits = filtered or hits[:1]

        if expand > 0:
            extra: List[SearchHit] = []
            for hit in hits:
                for neighbor in self.store.neighbors(hit.chunk, radius=expand):
                    extra.append(SearchHit(chunk=neighbor, score=hit.score * 0.85))
            hits.extend(extra)

        seen: set = set()
        unique: List[SearchHit] = []
        for hit in sorted(hits, key=lambda h: h.score, reverse=True):
            if hit.chunk.id in seen:
                continue
            seen.add(hit.chunk.id)
            unique.append(hit)
        return unique[:max_chunks]

    # ------------------------------------------------------------ 提示词
    def _history_messages(self, history: Optional[Iterable[Dict[str, str]]]) -> List[Dict[str, str]]:
        if not history:
            return []
        turns = int(self.config.get("app.history_turns", 3))
        cleaned: List[Dict[str, str]] = []
        for item in history:
            role = str(item.get("role", "user"))
            if role not in ("user", "assistant"):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            cleaned.append({"role": role, "content": content[:400]})
        return cleaned[-turns * 2 :]

    def build_messages(
        self,
        question: str,
        hits: Sequence[SearchHit],
        history: Optional[Iterable[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        max_chars = int(self.config.get("llm.max_context_chars", 6000))
        context, _ = format_context(hits, max_chars)
        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self._history_messages(history))
        user_content = (
            "【参考资料】\n"
            f"{context}\n\n"
            "【问题】\n"
            f"{question.strip()}\n\n"
            "请严格依据参考资料作答，并在关键结论后标注 [编号]。"
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    # ------------------------------------------------------------ 问答
    def answer(
        self,
        question: str,
        history: Optional[Iterable[Dict[str, str]]] = None,
    ) -> Answer:
        started = time.perf_counter()
        question = (question or "").strip()
        if not question:
            return Answer(question="", text="请输入一个问题。", used_llm=False)

        try:
            if self.store.is_empty():
                return Answer(question=question, text=NO_DOCUMENT_HINT, used_llm=False)
            hits = self.retrieve(question)
        except (EmbeddingError, VectorStoreError) as exc:
            logger.warning("检索失败：%s", exc)
            return Answer(
                question=question,
                text=f"检索阶段出错：{exc}\n\n请运行 `python -m localdocqa check` 排查环境。",
                used_llm=False,
            )

        if not hits:
            return Answer(question=question, text=NO_HIT_HINT, used_llm=False)

        # 先按上下文预算裁剪，保证"回答里出现的 [编号]" 与"界面上展示的来源"一一对应
        max_chars = int(self.config.get("llm.max_context_chars", 6000))
        _, sources = format_context(hits, max_chars)
        context_hits = hits[: len(sources)] if sources else hits[:1]

        used_llm = False
        note = ""
        text = ""
        client = self.llm
        if client is None:
            text = _extractive_answer(question, context_hits)
        else:
            try:
                text = client.chat(self.build_messages(question, context_hits, history))
                used_llm = True
            except LLMError as exc:
                logger.warning("模型调用失败，降级为离线提取：%s", exc)
                text = _extractive_answer(question, context_hits)
                note = _llm_failure_note(self.config, exc)

        return Answer(
            question=question,
            text=text + note,
            sources=sources,
            hits=list(hits),
            elapsed=time.perf_counter() - started,
            used_llm=used_llm,
        )

    def stream(
        self,
        question: str,
        history: Optional[Iterable[Dict[str, str]]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """产出事件：
        {"type": "sources", "sources": [...]}
        {"type": "token", "text": "..."}
        {"type": "done", "answer": Answer}
        {"type": "error", "message": "..."}
        """
        started = time.perf_counter()
        question = (question or "").strip()
        if not question:
            yield {"type": "error", "message": "请输入一个问题。"}
            return

        try:
            if self.store.is_empty():
                yield {"type": "done", "answer": Answer(question=question, text=NO_DOCUMENT_HINT, used_llm=False)}
                return
            hits = self.retrieve(question)
        except (EmbeddingError, VectorStoreError) as exc:
            yield {"type": "error", "message": f"检索阶段出错：{exc}"}
            return

        if not hits:
            yield {"type": "done", "answer": Answer(question=question, text=NO_HIT_HINT, used_llm=False)}
            return

        max_chars = int(self.config.get("llm.max_context_chars", 6000))
        _, sources = format_context(hits, max_chars)
        context_hits = hits[: len(sources)] if sources else hits[:1]
        yield {"type": "sources", "sources": sources}

        client = self.llm
        collected: List[str] = []
        used_llm = False
        if client is None:
            text = _extractive_answer(question, context_hits)
            collected.append(text)
            yield {"type": "token", "text": text}
        else:
            try:
                for piece in client.chat(self.build_messages(question, context_hits, history), stream=True):
                    collected.append(piece)
                    yield {"type": "token", "text": piece}
                used_llm = True
            except LLMError as exc:
                logger.warning("流式生成失败，降级为离线提取：%s", exc)
                text = _extractive_answer(question, context_hits) + _llm_failure_note(self.config, exc)
                collected = [text]
                yield {"type": "token", "text": text}

        yield {
            "type": "done",
            "answer": Answer(
                question=question,
                text="".join(collected),
                sources=sources,
                hits=list(hits),
                elapsed=time.perf_counter() - started,
                used_llm=used_llm,
            ),
        }
