"""命令行入口：python -m localdocqa <ingest|ask|search|stats|check>

示例：
    python -m localdocqa ingest examples/sample_docs --collection demo
    python -m localdocqa ask "这份文档讲了什么" --collection demo
    python -m localdocqa search "关键词" --collection demo -k 3
    python -m localdocqa stats
    python -m localdocqa check
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from typing import List, Optional

from . import __version__
from .config import Config
from .embeddings import EmbeddingError, build_embeddings, list_ollama_models
from .ingest import ingest, list_documents
from .llm import LLMError, build_llm
from .logging_utils import setup_logging
from .rag_chain import RAGEngine
from .vector_store import CollectionManager


def _build_config(args: argparse.Namespace) -> Config:
    return Config.load(path=getattr(args, "config", None))


def _print_sources(sources: List[dict]) -> None:
    if not sources:
        return
    print("\n来源：")
    for source in sources:
        print(
            f"  [{source['index']}] {source['citation']}"
            f"  (相似度 {source['score']}, chunk #{source['chunk_index']})"
        )


# ----------------------------------------------------------------- 子命令
def cmd_ingest(args: argparse.Namespace) -> int:
    config = _build_config(args)
    report = ingest(
        args.paths,
        collection=args.collection,
        config=config,
        force=args.force,
    )
    print(report.summary())
    for error in report.errors:
        print(f"  [失败] {error['file']}: {error['error']}", file=sys.stderr)
    return 0 if report.ok or report.new_files else 1


def cmd_ask(args: argparse.Namespace) -> int:
    config = _build_config(args)
    engine = RAGEngine(config, collection=args.collection)

    if not args.no_stream:
        printed = False
        for event in engine.stream(args.question):
            if event["type"] == "token":
                print(event["text"], end="", flush=True)
                printed = True
            elif event["type"] == "sources":
                _sources_cache = event["sources"]
            elif event["type"] == "done":
                answer = event["answer"]
                if not printed:
                    print(answer.text)
                print()
                _print_sources(answer.sources)
                print(f"\n(耗时 {answer.elapsed:.2f}s，{'模型生成' if answer.used_llm else '离线提取'})")
            elif event["type"] == "error":
                print(f"\n[错误] {event['message']}", file=sys.stderr)
                return 1
        return 0

    answer = engine.answer(args.question)
    if args.json:
        print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(answer.text)
        _print_sources(answer.sources)
        print(f"\n(耗时 {answer.elapsed:.2f}s，{'模型生成' if answer.used_llm else '离线提取'})")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    config = _build_config(args)
    engine = RAGEngine(config, collection=args.collection)
    hits = engine.retrieve(args.query, top_k=args.top_k)
    if args.json:
        print(
            json.dumps(
                [hit.to_source(index) for index, hit in enumerate(hits, start=1)],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not hits:
        print("没有命中任何 chunk。")
        return 1
    for index, hit in enumerate(hits, start=1):
        preview = " ".join(hit.chunk.text.split())[:160]
        print(f"[{index}] score={hit.score:.4f}  {hit.chunk.citation()}  chunk#{hit.chunk.chunk_index}")
        print(f"    {preview}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    config = _build_config(args)
    manager = CollectionManager(config.collections_dir())
    collections = manager.list_collections()
    if args.json:
        print(json.dumps(manager.stats(), ensure_ascii=False, indent=2))
        return 0
    if not collections:
        print(f"还没有任何集合。数据目录：{config.collections_dir()}")
        return 0
    print(f"数据目录：{config.collections_dir()}")
    print(f"{'集合':<20}{'文档':>6}{'chunk':>8}{'维度':>8}")
    for item in manager.stats():
        print(
            f"{item.get('name', ''):<20}{item.get('documents', 0):>6}"
            f"{item.get('chunks', 0):>8}{item.get('dim', 0):>8}"
        )
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    config = _build_config(args)
    ok = True
    print(f"LocalDocQA {__version__}")
    print(f"Python   : {platform.python_version()} ({sys.executable})")
    print(f"平台     : {platform.platform()}")
    print(f"配置文件 : {config.source or '未找到，使用内置默认值'}")
    print(f"数据目录 : {config.collections_dir()}")

    print("\n依赖检查：")
    for module, package in (
        ("numpy", "numpy"),
        ("yaml", "pyyaml"),
        ("requests", "requests"),
        ("pypdf", "pypdf"),
        ("docx", "python-docx"),
        ("streamlit", "streamlit"),
    ):
        try:
            __import__(module)
            print(f"  [OK]   {package}")
        except ImportError:
            level = "必装" if package != "streamlit" else "仅在网页界面需要"
            print(f"  [缺失] {package}（{level}）")
            if package != "streamlit":
                ok = False

    print("\n模型服务：")
    llm_cfg = config.section("llm")
    client = build_llm(llm_cfg)
    if client is None:
        print("  [提示] llm.provider = none，将使用离线提取模式")
    else:
        healthy, message = client.health_check()
        print(f"  [{'OK' if healthy else '警告'}] {client.name}: {message}")
        if not healthy:
            ok = False

    emb_cfg = config.section("embeddings")
    print("\n向量化：")
    try:
        embedder = build_embeddings(emb_cfg)
        probe = embedder.embed_query("连通性测试")
        print(f"  [OK]   {embedder.name}，维度 {len(probe)}")
    except EmbeddingError as exc:
        print(f"  [失败] {exc}")
        ok = False

    models = list_ollama_models(str(emb_cfg.get("base_url", "http://127.0.0.1:11434")))
    print(f"  已安装模型：{', '.join(models) if models else '未检测到 Ollama 服务'}")

    print("\n集合：")
    for item in CollectionManager(config.collections_dir()).stats():
        print(f"  - {item.get('name')}: {item.get('documents')} 文档 / {item.get('chunks')} chunk")

    print("\n结论：", "环境可用" if ok else "存在需要处理的问题，请按上面的提示修复")
    return 0 if ok else 1


def cmd_list(args: argparse.Namespace) -> int:
    config = _build_config(args)
    documents = list_documents(args.collection, config)
    if args.json:
        print(json.dumps(documents, ensure_ascii=False, indent=2))
        return 0
    if not documents:
        print("该集合还没有文档。")
        return 0
    for document in documents:
        print(
            f"{document.get('filename'):<32}{str(document.get('chunks')):>6} chunk"
            f"   {document.get('ingested_at', '')}"
        )
    return 0


# ----------------------------------------------------------------- 解析
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localdocqa",
        description="LocalDocQA —— 完全本地运行的文档问答工具",
    )
    parser.add_argument("--config", help="配置文件路径（默认 config.yaml）")
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="导入文件或目录")
    ingest_parser.add_argument("paths", nargs="+", help="文件或目录路径")
    ingest_parser.add_argument("-c", "--collection", default="default", help="集合名称")
    ingest_parser.add_argument("-f", "--force", action="store_true", help="忽略重复，强制重新入库")
    ingest_parser.set_defaults(func=cmd_ingest)

    ask_parser = subparsers.add_parser("ask", help="提问")
    ask_parser.add_argument("question", help="问题文本")
    ask_parser.add_argument("-c", "--collection", default="default", help="集合名称")
    ask_parser.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    ask_parser.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    ask_parser.set_defaults(func=cmd_ask)

    search_parser = subparsers.add_parser("search", help="只做检索，不生成")
    search_parser.add_argument("query", help="查询文本")
    search_parser.add_argument("-c", "--collection", default="default", help="集合名称")
    search_parser.add_argument("-k", "--top-k", type=int, default=5, help="返回条数")
    search_parser.add_argument("--json", action="store_true")
    search_parser.set_defaults(func=cmd_search)

    stats_parser = subparsers.add_parser("stats", help="查看所有集合的统计信息")
    stats_parser.add_argument("--json", action="store_true")
    stats_parser.set_defaults(func=cmd_stats)

    list_parser = subparsers.add_parser("list", help="列出某集合的文档")
    list_parser.add_argument("-c", "--collection", default="default")
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(func=cmd_list)

    check_parser = subparsers.add_parser("check", help="环境自检")
    check_parser.set_defaults(func=cmd_check)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")

    # 子命令自己的 -c 参数与全局 --config 互不冲突
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except (LLMError, EmbeddingError, FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
