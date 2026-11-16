"""大模型客户端：Ollama（默认）/ OpenAI 兼容接口，统一 chat/stream 接口。

设计要点：
- 只用 requests，不引入 openai / langchain 等重型依赖，安装体积小、可控；
- 流式输出统一成"逐段 yield 字符串"，UI 和 CLI 共用；
- 失败一律抛 ``LLMError``（带可执行建议），由上层决定降级策略。
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Optional, Tuple

import requests

from .logging_utils import get_logger

logger = get_logger("localdocqa.llm")

Message = Dict[str, str]


class LLMError(RuntimeError):
    """模型调用失败。"""


class LLMClient:
    name: str = "base"

    def chat(self, messages: List[Message], stream: bool = False) -> str | Iterator[str]:
        raise NotImplementedError

    def health_check(self) -> Tuple[bool, str]:
        raise NotImplementedError


class OllamaLLM(LLMClient):
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.2,
        num_ctx: int = 4096,
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def _payload(self, messages: List[Message], stream: bool) -> Dict:
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }

    def _timeout(self) -> Tuple[float, float]:
        """(连接, 读取)：本机服务未启动时 1 秒内失败，避免界面长时间无响应。"""
        return (1.0, self.timeout)

    def chat(self, messages: List[Message], stream: bool = False) -> str | Iterator[str]:
        if stream:
            return self._stream(messages)
        return self._complete(messages)

    def _complete(self, messages: List[Message]) -> str:
        url = f"{self.base_url}/api/chat"
        try:
            response = requests.post(url, json=self._payload(messages, False), timeout=self._timeout())
        except requests.RequestException as exc:
            raise LLMError(f"连接 Ollama 失败（{url}）：{exc}") from exc
        if response.status_code == 404:
            raise LLMError(f"模型 {self.model} 不存在，请执行 `ollama pull {self.model}`")
        if response.status_code >= 400:
            raise LLMError(f"Ollama 返回 {response.status_code}：{response.text[:200]}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMError("Ollama 返回内容不是合法 JSON") from exc
        content = (payload.get("message") or {}).get("content", "")
        if not content:
            raise LLMError("Ollama 返回空内容（可尝试减小上下文长度或更换模型）")
        return content

    def _stream(self, messages: List[Message]) -> Iterator[str]:
        url = f"{self.base_url}/api/chat"
        try:
            response = requests.post(
                url, json=self._payload(messages, True), timeout=self._timeout(), stream=True
            )
        except requests.RequestException as exc:
            raise LLMError(f"连接 Ollama 失败（{url}）：{exc}") from exc
        if response.status_code >= 400:
            raise LLMError(
                f"Ollama 返回 {response.status_code}：{response.text[:200]}。"
                f"请确认已执行 `ollama pull {self.model}`"
            )
        got_content = False
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except ValueError:
                continue
            if payload.get("error"):
                raise LLMError(f"Ollama 流式返回错误：{payload['error']}")
            piece = (payload.get("message") or {}).get("content", "")
            if piece:
                got_content = True
                yield piece
            if payload.get("done"):
                break
        if not got_content:
            raise LLMError("模型流式输出为空，请检查模型是否正常加载")

    def health_check(self) -> Tuple[bool, str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=(1.0, 8))
            response.raise_for_status()
            models = [m.get("name", "") for m in response.json().get("models", [])]
        except (requests.RequestException, ValueError) as exc:
            return False, f"无法连接 Ollama（{self.base_url}）：{exc}"
        # Ollama 的名称可能带 :latest 后缀，用前缀匹配
        if any(m.split(":")[0] == self.model.split(":")[0] for m in models):
            return True, f"{self.model} 已就绪"
        return False, f"未找到模型 {self.model}，已安装：{', '.join(models) or '无'}"


class OpenAICompatLLM(LLMClient):
    """任何 OpenAI 兼容网关（vLLM / LM Studio / DeepSeek / 自建代理）都可接。"""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.2,
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.temperature = temperature
        self.timeout = timeout
        self.name = f"openai:{model}"

    def _endpoint(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat(self, messages: List[Message], stream: bool = False) -> str | Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream,
        }
        try:
            response = requests.post(
                self._endpoint(),
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
                stream=stream,
            )
        except requests.RequestException as exc:
            raise LLMError(f"连接模型服务失败（{self._endpoint()}）：{exc}") from exc
        if response.status_code >= 400:
            raise LLMError(f"模型服务返回 {response.status_code}：{response.text[:200]}")
        if not stream:
            try:
                content = response.json()["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError) as exc:
                raise LLMError("模型返回格式异常，期望 OpenAI chat.completions 结构") from exc
            return content or ""

        def _iterator() -> Iterator[str]:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data = raw_line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                    piece = payload["choices"][0]["delta"].get("content", "")
                except (ValueError, KeyError, IndexError):
                    continue
                if piece:
                    yield piece

        return _iterator()

    def health_check(self) -> Tuple[bool, str]:
        if not self.api_key:
            return False, "未设置 API Key 环境变量"
        return True, "已配置 OpenAI 兼容接口（未做真实请求校验）"


def build_llm(llm_cfg: Dict) -> Optional[LLMClient]:
    """provider=none 时返回 None，上层走离线提取模式。"""
    provider = str(llm_cfg.get("provider", "ollama")).lower()
    if provider in ("none", "off", "extractive", ""):
        return None
    if provider == "ollama":
        return OllamaLLM(
            model=str(llm_cfg.get("model", "qwen2.5:7b")),
            base_url=str(llm_cfg.get("base_url", "http://127.0.0.1:11434")),
            temperature=float(llm_cfg.get("temperature", 0.2)),
            num_ctx=int(llm_cfg.get("num_ctx", 4096)),
            timeout=float(llm_cfg.get("timeout", 180)),
        )
    if provider in ("openai", "openai_compatible", "openai-compatible"):
        return OpenAICompatLLM(
            model=str(llm_cfg.get("model", "gpt-4o-mini")),
            base_url=str(llm_cfg.get("base_url", "http://127.0.0.1:8000/v1")),
            api_key_env=str(llm_cfg.get("api_key_env", "OPENAI_API_KEY")),
            temperature=float(llm_cfg.get("temperature", 0.2)),
            timeout=float(llm_cfg.get("timeout", 180)),
        )
    raise ValueError(f"未知的 llm.provider：{provider}（可选 ollama / openai / none）")
