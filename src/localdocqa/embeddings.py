"""向量化（Embedding）：Ollama 语义向量 + 离线哈希向量两套实现。

- ``ollama``：调用本地 Ollama 的 embedding 接口（推荐，如 nomic-embed-text，768 维）
- ``hash``  ：不依赖任何模型/网络的词法向量，用于离线演示、单元测试和 CI

两者输出都已做 L2 归一化，因此向量库直接用点积即余弦相似度。
题目/接口设计成同一份 ``EmbeddingProvider``，换模型不需要改上层代码。
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

from .logging_utils import get_logger

logger = get_logger("localdocqa.embeddings")

_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[a-zA-Z0-9_]{2,}")


class EmbeddingError(RuntimeError):
    """向量化失败（模型未拉取、服务不可用、维度异常等）。"""


class EmbeddingProvider:
    name: str = "base"
    # semantic = 语义向量（分数分布较集中）；lexical = 词法向量（分数整体偏低）
    kind: str = "semantic"
    dim: Optional[int] = None

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class HashingEmbeddings(EmbeddingProvider):
    """离线词法向量（signed hashing）。

    中文取"单字 + 相邻二字"，英文取长度≥2的单词；词频用 1+log(count) 压制长文本优势。
    它不是语义模型——同义不同词召回不到，但在断网/无模型的场景下能保证整条链路可用，
    也让 CI 不必下载任何模型即可跑通端到端测试。
    """

    name = "hash"
    kind = "lexical"

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim 必须为正整数")
        self.dim = int(dim)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """中文以二字组合为主特征（单字的区分度太低，只在整个文本很短时才使用），
        英文/数字取长度≥2的单词。"""
        lowered = text.lower()
        tokens: List[str] = []
        chars = _CJK.findall(lowered)
        if len(chars) >= 2:
            tokens.extend("".join(pair) for pair in zip(chars, chars[1:], strict=False))
        elif chars:
            tokens.extend(chars)
        tokens.extend(_WORD.findall(lowered))
        return tokens

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = Counter(self._tokenize(text or ""))
            for token, count in counts.items():
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, index] += sign * (1.0 + math.log1p(count))
        # 全零行给一个常量，避免归一化后出现 NaN 并让检索拿到 0 分
        empty_rows = np.linalg.norm(matrix, axis=1) == 0
        matrix[empty_rows, 0] = 1.0
        return self._normalize(matrix)


class OllamaEmbeddings(EmbeddingProvider):
    """Ollama embedding 客户端。

    优先调用新版 ``/api/embed``（支持批量），若模型/版本不支持则回退到
    ``/api/embeddings``（单条）。批量失败会逐条重试，避免一条异常拖垮整批。
    """

    name = "ollama"

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://127.0.0.1:11434",
        batch_size: int = 16,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = max(1, int(batch_size))
        self.timeout = timeout
        self.dim: Optional[int] = None

    # ------------------------------------------------------------ 低层请求
    def _post(self, path: str, payload: Dict) -> Dict:
        url = f"{self.base_url}{path}"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise EmbeddingError(f"连接 Ollama 失败（{url}）：{exc}") from exc
        if response.status_code == 404:
            raise EmbeddingError(f"Ollama 接口不存在：{url}")
        if response.status_code >= 400:
            raise EmbeddingError(
                f"Ollama 返回 {response.status_code}：{response.text[:200]}。"
                f"请确认已执行 `ollama pull {self.model}`"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise EmbeddingError("Ollama 返回内容不是合法 JSON") from exc

    def _embed_batch_new_api(self, texts: Sequence[str]) -> np.ndarray:
        payload = self._post("/api/embed", {"model": self.model, "input": list(texts)})
        embeddings = payload.get("embeddings")
        if not embeddings:
            raise EmbeddingError(f"/api/embed 未返回 embeddings，模型={self.model}")
        return np.asarray(embeddings, dtype=np.float32)

    def _embed_single_legacy(self, text: str) -> np.ndarray:
        payload = self._post("/api/embeddings", {"model": self.model, "prompt": text})
        embedding = payload.get("embedding")
        if not embedding:
            raise EmbeddingError(f"/api/embeddings 未返回 embedding，模型={self.model}")
        return np.asarray(embedding, dtype=np.float32)

    # ------------------------------------------------------------ 对外接口
    def _embed_chunk(self, texts: Sequence[str]) -> np.ndarray:
        try:
            matrix = self._embed_batch_new_api(texts)
        except EmbeddingError:
            matrix = np.vstack([self._embed_single_legacy(text) for text in texts])
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise EmbeddingError(f"向量数量与文本数量不一致：{matrix.shape} vs {len(texts)}")
        if self.dim is None:
            self.dim = int(matrix.shape[1])
        elif matrix.shape[1] != self.dim:
            raise EmbeddingError(f"向量维度不一致：期望 {self.dim}，实际 {matrix.shape[1]}")
        return self._normalize(matrix)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        batches = [
            list(texts[i : i + self.batch_size]) for i in range(0, len(texts), self.batch_size)
        ]
        return np.vstack([self._embed_chunk(batch) for batch in batches])

    def health_check(self) -> Tuple[bool, str]:
        try:
            self.embed_query("连通性测试")
            return True, f"模型 {self.model} 可用（{self.dim} 维）"
        except EmbeddingError as exc:
            return False, str(exc)


def list_ollama_models(base_url: str = "http://127.0.0.1:11434", timeout: float = 10.0) -> List[str]:
    """列出本地 Ollama 已拉取的模型名，服务不可用时返回空列表。"""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []
    return [str(item.get("name", "")) for item in payload.get("models", []) if item.get("name")]


def build_embeddings(embeddings_cfg: Dict, fallback_log: bool = True) -> EmbeddingProvider:
    """按配置构造 embedding 提供者；ollama 不可用时按配置回退到离线向量。"""
    provider = str(embeddings_cfg.get("provider", "ollama")).lower()
    if provider == "hash":
        return HashingEmbeddings(dim=int(embeddings_cfg.get("dim", 256)))

    ollama = OllamaEmbeddings(
        model=str(embeddings_cfg.get("model", "nomic-embed-text")),
        base_url=str(embeddings_cfg.get("base_url", "http://127.0.0.1:11434")),
        batch_size=int(embeddings_cfg.get("batch_size", 16)),
    )
    ok, message = ollama.health_check()
    if ok:
        return ollama

    if bool(embeddings_cfg.get("fallback_to_hash", True)):
        if fallback_log:
            logger.warning("Embedding 服务不可用，已回退到离线哈希向量：%s", message)
        return HashingEmbeddings(dim=int(embeddings_cfg.get("dim", 256)))
    raise EmbeddingError(message)
