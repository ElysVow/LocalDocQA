"""端到端冒烟测试：不依赖任何模型，用离线向量跑通"入库 → 检索 → 回答"。

CI 里也会跑这个脚本，用来确认主链路没有被改坏。
用法：
    python scripts/smoke_test.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from localdocqa.config import Config  # noqa: E402
from localdocqa.embeddings import HashingEmbeddings, build_embeddings  # noqa: E402
from localdocqa.ingest import ingest  # noqa: E402
from localdocqa.rag_chain import RAGEngine  # noqa: E402


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="localdocqa-smoke-"))
    try:
        config = Config({}, root=workdir)
        config.set("llm.provider", "none")          # 走离线摘录模式
        config.set("chunking.size", 300)
        config.set("chunking.overlap", 40)
        config.set("retrieval.score_threshold", -1.0)
        config.save(workdir / "config.yaml")

        embeddings = HashingEmbeddings(dim=256)
        report = ingest([str(ROOT / "examples" / "sample_docs")], collection="smoke", config=config,
                        embeddings=embeddings)
        print(f"入库：{report.summary()}")
        assert report.new_files == 3, "示例文档应全部入库"
        assert report.chunks > 0

        engine = RAGEngine(config, collection="smoke", embeddings=embeddings)
        answer = engine.answer("TCP 三次握手的三个报文分别是什么")
        print("\n回答：")
        print(answer.text[:400])
        print("\n来源：", [s["citation"] for s in answer.sources])
        assert answer.sources, "应至少召回一段文档"
        assert "SYN" in "".join(hit.chunk.text for hit in answer.hits), "应召回含 SYN 的段落"

        # 二次入库应全部被判定为重复
        second = ingest([str(ROOT / "examples" / "sample_docs")], collection="smoke", config=config,
                        embeddings=embeddings)
        assert second.new_files == 0 and second.skipped_files == 3

        # 配置为 ollama 但服务不可用时，应自动回退而不是崩溃
        fallback_cfg = Config.load(root=workdir)
        fallback_embedder = build_embeddings(fallback_cfg.section("embeddings"))
        print(f"\n降级检查：provider={fallback_cfg.get('embeddings.provider')} -> {fallback_embedder.name}")

        print("\n冒烟测试通过。")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
