from __future__ import annotations

import numpy as np
import pytest

from localdocqa.config import Config
from localdocqa.embeddings import EmbeddingProvider, HashingEmbeddings
from localdocqa.ingest import ingest
from localdocqa.llm import LLMClient
from localdocqa.models import Chunk
from localdocqa.rag_chain import NO_DOCUMENT_HINT, NO_HIT_HINT, RAGEngine, format_context
from localdocqa.vector_store import CollectionManager


class FakeLLM(LLMClient):
    """记录收到的 messages，用于验证 prompt 里确实带上了检索到的资料。"""

    name = "fake"

    def __init__(self) -> None:
        self.calls = []

    def chat(self, messages, stream=False):
        self.calls.append(messages)
        if stream:
            return iter(["根据资料 ", "[1]，验收标准包含三条。"])
        return "根据资料 [1]，验收标准包含功能可用、文档齐全、测试通过三条。"

    def health_check(self):
        return True, "fake"


class OrthogonalEmbeddings(EmbeddingProvider):
    """语义型 + 与库中向量正交：用来稳定地验证"分数低于阈值 → 不命中"。"""

    name = "orthogonal"
    kind = "semantic"
    dim = 2

    def embed_documents(self, texts):
        return self._normalize(np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (len(texts), 1)))

    def embed_query(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)


@pytest.fixture()
def config(tmp_path):
    cfg = Config({}, root=tmp_path)
    cfg.set("llm.provider", "none")  # 默认走离线提取，测试不依赖任何模型
    cfg.set("chunking.size", 200)
    cfg.set("chunking.overlap", 20)
    cfg.set("retrieval.score_threshold", -1.0)
    return cfg


@pytest.fixture()
def docs(config, tmp_path):
    directory = tmp_path / "docs"
    directory.mkdir()
    (directory / "spec.md").write_text(
        "# 项目验收\n\n" + "本项目的验收标准包括三条：功能可用、文档齐全、测试通过。" * 5,
        encoding="utf-8",
    )
    (directory / "net.md").write_text(
        "计算机网络笔记：TCP 三次握手包含 SYN、SYN+ACK、ACK 三个报文，用于建立可靠连接。",
        encoding="utf-8",
    )
    return directory


@pytest.fixture()
def engine(config, docs):
    embeddings = HashingEmbeddings()
    ingest([str(docs)], collection="demo", config=config, embeddings=embeddings)
    return RAGEngine(config, collection="demo", embeddings=embeddings)


def test_answer_offline_returns_sources(engine):
    answer = engine.answer("项目的验收标准是什么")
    assert answer.text
    assert "离线提取" in answer.text
    assert answer.sources
    assert answer.sources[0]["filename"] == "spec.md"
    assert answer.hits


def test_answer_without_documents(config):
    engine = RAGEngine(config, collection="empty", embeddings=HashingEmbeddings())
    answer = engine.answer("随便问点什么")
    assert answer.text == NO_DOCUMENT_HINT
    assert answer.sources == []


def test_answer_without_hits(config):
    """语义模式下，最高分低于阈值时应返回"没检索到"，而不是硬凑答案。"""
    manager = CollectionManager(config.collections_dir())
    store = manager.ensure("far")
    embedder = OrthogonalEmbeddings()
    store.add(
        [Chunk(id="far-1", text="完全无关的内容", source="/tmp/far.md", doc_id="far", chunk_index=0)],
        embedder.embed_documents(["完全无关的内容"]),
    )
    store.save()

    config.set("retrieval.score_threshold", 0.3)
    engine = RAGEngine(config, collection="far", embeddings=embedder)
    answer = engine.answer("量子纠缠与超导材料")
    assert answer.text == NO_HIT_HINT
    assert answer.sources == []


def test_answer_uses_llm_when_available(engine):
    fake = FakeLLM()
    engine._llm = fake
    engine._llm_ready = True
    answer = engine.answer("验收标准")
    assert answer.used_llm is True
    assert "验收标准" in answer.text
    messages = fake.calls[0]
    assert messages[0]["role"] == "system"
    assert "【参考资料】" in messages[-1]["content"]
    assert "[1]" in messages[-1]["content"]


def test_stream_events(engine):
    events = list(engine.stream("TCP 三次握手"))
    types = [event["type"] for event in events]
    assert types[0] == "sources"
    assert "done" in types
    done = events[-1]["answer"]
    assert done.sources
    assert "离线提取" in done.text


def test_stream_error_on_empty_question(engine):
    events = list(engine.stream("   "))
    assert events[-1]["type"] == "error"


def test_format_context_numbers_match_sources(engine):
    hits = engine.retrieve("验收标准")
    context, sources = format_context(hits, max_chars=10_000)
    for source in sources:
        assert f"[{source['index']}]" in context


def test_retrieve_returns_ranked_hits(engine):
    hits = engine.retrieve("TCP 三次握手", top_k=3)
    assert hits
    assert hits[0].score >= hits[-1].score


def test_history_is_trimmed(engine):
    fake = FakeLLM()
    engine._llm = fake
    engine._llm_ready = True
    history = [{"role": "user", "content": f"历史问题{i}"} for i in range(10)]
    engine.answer("验收标准", history=history)
    messages = fake.calls[0]
    # system + 最近 3 轮(6 条) + 当前提问
    assert len(messages) == 1 + 6 + 1
    assert messages[1]["content"] == "历史问题4"
