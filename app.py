"""Streamlit 网页界面：上传入库、对话问答、来源溯源、文档与集合管理。

启动：
    streamlit run app.py

两个布局约束：
1. 页面切换用主区域横向单选，而不是 st.tabs —— st.chat_input 只有在主区域
   （不在 tab/container 内）才会固定在视口底部，放进 tab 会变成行内元素；
2. 页面级状态（对话、上次入库报告）都放在 st.session_state，并落盘到
   data/conversations/<集合>.json，刷新或重启后还能接着聊。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# 以 `streamlit run app.py` 启动时，src 目录不在 sys.path 中
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st  # noqa: E402

from localdocqa.archive import export_collection, import_collection  # noqa: E402
from localdocqa.config import Config  # noqa: E402
from localdocqa.conversations import (  # noqa: E402
    clear_conversation,
    conversations_dir,
    load_conversation,
    save_conversation,
)
from localdocqa.embeddings import EmbeddingError, build_embeddings, list_ollama_models  # noqa: E402
from localdocqa.ingest import IngestReport, ingest, list_documents, remove_document  # noqa: E402
from localdocqa.llm import build_llm  # noqa: E402
from localdocqa.rag_chain import RAGEngine  # noqa: E402
from localdocqa.text_utils import highlight  # noqa: E402
from localdocqa.vector_store import CollectionManager, VectorStoreError  # noqa: E402

SAMPLE_DIR = ROOT / "examples" / "sample_docs"
UPLOAD_TYPES = ["pdf", "docx", "md", "markdown", "txt", "log", "csv", "json", "html", "htm"]
PAGES = ["问答", "文档", "设置", "关于"]

CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3.5rem; max-width: 1180px;}
h1 {font-size: 1.9rem !important; letter-spacing: -0.02em; margin-bottom: 0.15rem;}
h2 {font-size: 1.45rem !important; margin-top: 1.8rem;}
h3 {font-size: 1.05rem !important; letter-spacing: 0.01em;}
[data-testid="stMarkdownContainer"] p {line-height: 1.75;}
[data-testid="stCaptionContainer"] {color: #7b7b7b;}
[data-testid="stSidebar"] h3 {font-size: 0.92rem !important; color: #4c4c4c;}
[data-testid="stChatMessage"] {padding: 0.4rem 0;}
[data-testid="stChatInput"] textarea {font-size: 0.95rem;}
mark {background: #ffe58f; color: inherit; padding: 0 2px; border-radius: 3px;}
.stButton > button {border-radius: 8px;}
div[data-testid="stDownloadButton"] > button {border-radius: 8px;}
hr {margin: 1.2rem 0;}
</style>
"""


# ------------------------------------------------------------------ 通用工具
def short_path(source: object) -> str:
    """表格里显示相对路径，绝对路径太长会把列撑爆。"""
    if not source:
        return "-"
    path = Path(str(source))
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path.name


def format_size(size: object) -> str:
    try:
        value = float(size or 0)
    except (TypeError, ValueError):
        return "-"
    if value <= 0:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return "-"


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def file_type_label(filename: object) -> str:
    suffix = Path(str(filename or "")).suffix.lower().lstrip(".")
    return suffix.upper() if suffix else "未知"


def messages_to_markdown(messages: list[dict], collection: str) -> str:
    lines = [
        f"# {collection} 问答记录",
        "",
        f"导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for message in messages:
        role = "我" if message["role"] == "user" else "LocalDocQA"
        lines.extend([f"## {role}", "", message["content"]])
        sources = message.get("sources") or []
        if sources:
            lines.extend(["", "**来源**"])
            for source in sources:
                lines.append(f"- [{source['index']}] {source['citation']}（相似度 {source['score']}）")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ 缓存
@st.cache_resource(show_spinner=False)
def cached_embeddings(signature: str):
    """按配置签名缓存 embedding 客户端，避免每次交互都去探测模型。"""
    return build_embeddings(Config.load().section("embeddings"))


@st.cache_data(show_spinner=False, ttl=300)
def cached_models(base_url: str):
    return list_ollama_models(base_url)


@st.cache_data(show_spinner=False, ttl=120)
def cached_llm_status(provider: str, base_url: str, model: str) -> tuple[bool, str]:
    """模型状态探测带缓存：否则每次点击都发一次请求，界面会明显卡顿。"""
    client = build_llm({"provider": provider, "base_url": base_url, "model": model})
    if client is None:
        return True, "未启用（离线提取模式）"
    return client.health_check()


# ------------------------------------------------------------------ 引擎与数据
def collection_options() -> list[str]:
    existing = CollectionManager(Config.load().collections_dir()).list_collections()
    return existing or ["default"]


def build_engine(collection: str, config: Config) -> RAGEngine:
    signature = json.dumps(config.section("embeddings"), sort_keys=True)
    return RAGEngine(config, collection=collection, embeddings=cached_embeddings(signature))


def do_ingest(paths: list[str], collection: str, config: Config, force: bool = False) -> IngestReport:
    progress_bar = st.progress(0.0, text="准备中…")

    def on_progress(message: str, fraction: float | None) -> None:
        progress_bar.progress(min(max(fraction or 0.0, 0.0), 1.0), text=message)

    report = ingest(paths, collection=collection, config=config, force=force, progress=on_progress)
    progress_bar.empty()
    return report


def render_report(report: IngestReport | None, details: bool = True) -> None:
    if not report:
        return
    if report.errors:
        st.warning(report.summary())
    else:
        st.success(report.summary())
    if details and report.details:
        with st.expander("逐文件结果"):
            st.dataframe(
                [
                    {
                        "文件": item.get("file"),
                        "结果": item.get("status"),
                        "chunk": item.get("chunks"),
                        "说明": item.get("reason") or "",
                    }
                    for item in report.details
                ],
                use_container_width=True,
                hide_index=True,
            )


# ------------------------------------------------------------------ 来源展示
def render_sources(sources: list[dict], query: str = "") -> None:
    if not sources:
        return
    with st.expander(f"参考来源（{len(sources)} 段）", expanded=False):
        for source in sources:
            with st.container(border=True):
                st.markdown(f"**[{source['index']}] {source['citation']}**")
                st.markdown(
                    f"<span style='color:#8a8a8a;font-size:12px'>"
                    f"相似度 {source['score']} ｜ chunk #{source['chunk_index']}"
                    f"</span>",
                    unsafe_allow_html=True,
                )
                body = highlight(source.get("preview") or "", query)
                st.markdown(
                    f"<div style='font-size:13px;line-height:1.7;color:#5a5a5a;"
                    f"border-left:3px solid #d9dee6;padding:2px 0 2px 10px;margin:6px 0 4px 0'>"
                    f"{body}</div>",
                    unsafe_allow_html=True,
                )
                full_text = source.get("text") or source.get("preview") or ""
                with st.popover("查看原文"):
                    st.code(full_text, language=None)


def render_answer(text: str) -> None:
    st.markdown(text)
    with st.popover("复制回答"):
        st.code(text, language=None)


def render_answer_meta(elapsed: float, used_llm: bool) -> None:
    st.caption(
        f"耗时 {elapsed:.1f}s ｜ "
        f"{'模型生成' if used_llm else '离线提取（未接入大模型）'}"
    )


# ------------------------------------------------------------------ 页面：问答
def example_questions(collection: str, config: Config) -> list[str]:
    documents = list_documents(collection, config)
    questions: list[str] = []
    if documents:
        questions.append(f"《{documents[0]['filename']}》主要讲了什么？")
    questions.append("这批资料的核心结论是什么？")
    if collection == "demo":
        questions.append("实验报告要交什么？")
    else:
        questions.append("有哪些提交要求和截止时间？")
    return questions[:3]


def render_examples(collection: str, config: Config, expanded: bool) -> None:
    """示例区常驻页面：空对话时展开做三步引导，有对话后收成一行。

    常驻而不是"有消息就删掉"，是因为 Streamlit 在流式生成期间不会立即清理
    上一轮末尾的元素，删掉会在回答下方残留一排示例按钮。
    """
    with st.expander("示例问题", expanded=expanded):
        if expanded:
            st.markdown("#### 三步开始")
            steps = [
                ("① 导入资料", "左侧上传 PDF / Word / Markdown，或点「导入示例文档」"),
                ("② 自动建索引", "切分 → 向量化 → 存入本机，重复文件按内容哈希跳过"),
                ("③ 提问并溯源", "回答里的 [1] [2] 对应下方来源，可展开看原文"),
            ]
            for column, (title, description) in zip(st.columns(3), steps, strict=False):
                with column:
                    with st.container(border=True):
                        st.markdown(f"**{title}**")
                        st.caption(description)
            st.markdown("#### 直接试试")

        questions = example_questions(collection, config)
        for column, question in zip(st.columns(len(questions)), questions, strict=False):
            with column:
                if st.button(question, key=f"example-{question}", use_container_width=True):
                    st.session_state.pending_prompt = question
                    st.rerun()


def render_chat(config: Config, collection: str, top_k: int, stream_mode: bool) -> None:
    manager = CollectionManager(config.collections_dir())
    stats = manager.get(collection).stats()
    left, middle, right = st.columns([3, 1, 1])
    with left:
        st.caption(
            f"当前集合 **{collection}**：{stats.get('documents', 0)} 个文档 / "
            f"{stats.get('chunks', 0)} 个 chunk"
        )
    if st.session_state.messages:
        with middle:
            st.download_button(
                "导出对话",
                data=messages_to_markdown(st.session_state.messages, collection),
                file_name=f"{collection}-qa-{datetime.now().strftime('%Y%m%d-%H%M')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with right:
            with st.popover("清空对话", use_container_width=True):
                st.caption("清空后无法恢复，确认要删除当前对话吗？")
                if st.button("确认清空", key="confirm-clear-chat"):
                    st.session_state.messages = []
                    clear_conversation(collection, config)
                    st.rerun()

    render_examples(collection, config, expanded=not st.session_state.messages)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer(message["content"])
            else:
                st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"], message.get("query", ""))

    if (
        st.session_state.messages
        and st.session_state.messages[-1]["role"] == "assistant"
        and len(st.session_state.messages) >= 2
    ):
        if st.button("重新生成上一条回答", key="regenerate"):
            last_question = st.session_state.messages[-2]["content"]
            st.session_state.messages.pop()
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()
            save_conversation(collection, st.session_state.messages, config)
            st.session_state.pending_prompt = last_question
            st.rerun()

    # 第二步：上一轮已经把提问写入 messages，这里只负责生成回答。
    # 分成两步是为了先让空态引导退场，避免回答流式输出时还残留示例卡片。
    request = st.session_state.pop("pending_answer", None)
    if request:
        engine = build_engine(collection, config)
        history = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.messages[:-1]
        ]
        with st.chat_message("assistant"):
            if stream_mode:
                placeholder = st.empty()
                placeholder.markdown("*正在检索文档…*")
                buffer: list[str] = []
                sources: list[dict] = []
                meta = {"used_llm": False, "elapsed": 0.0}
                for event in engine.stream(request, history=history):
                    if event["type"] == "token":
                        buffer.append(event["text"])
                        placeholder.markdown("".join(buffer) + "▌")
                    elif event["type"] == "sources":
                        sources = event["sources"]
                    elif event["type"] == "done":
                        answer = event["answer"]
                        meta = {"used_llm": answer.used_llm, "elapsed": answer.elapsed}
                        sources = answer.sources
                        placeholder.markdown(answer.text)
                        buffer = [answer.text]
                    elif event["type"] == "error":
                        placeholder.error(event["message"])
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"出错了：{event['message']}"}
                        )
                        save_conversation(collection, st.session_state.messages, config)
                        st.stop()
                answer_text = "".join(buffer)
                render_answer_meta(meta["elapsed"], meta["used_llm"])
                render_sources(sources, request)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer_text, "sources": sources, "query": request}
                )
            else:
                answer = engine.answer(request, history=history)
                render_answer(answer.text)
                render_answer_meta(answer.elapsed, answer.used_llm)
                render_sources(answer.sources, request)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer.text,
                        "sources": answer.sources,
                        "query": request,
                    }
                )
        save_conversation(collection, st.session_state.messages, config)
        # 重跑一次，让“重新生成 / 清空 / 导出”等依赖消息状态的控件立即出现
        st.rerun()

    pending = st.session_state.pop("pending_prompt", None)
    typed = st.chat_input("针对文档提问，例如：这份文档的核心结论是什么？")
    prompt = pending or typed
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_conversation(collection, st.session_state.messages, config)
        st.session_state.pending_answer = prompt
        st.rerun()


# ------------------------------------------------------------------ 页面：文档
def render_document_detail(config: Config, collection: str, document: dict) -> None:
    store = CollectionManager(config.collections_dir()).get(collection)
    doc_id = str(document.get("doc_id"))
    tabs = st.tabs(["内容预览", "文件信息", "删除"])
    with tabs[0]:
        chunks = store.chunks_for_doc(doc_id)
        st.caption(f"共 {len(chunks)} 段，下面展示前 6 段")
        for chunk in chunks[:6]:
            with st.container(border=True):
                page = f" · 第 {chunk.page} 页" if chunk.page else ""
                st.caption(f"chunk #{chunk.chunk_index}{page}")
                text = chunk.text.strip()
                st.markdown(text[:900] + ("…" if len(text) > 900 else ""))
        if len(chunks) > 6:
            st.info(f"还有 {len(chunks) - 6} 段未展示，可用命令行 `localdocqa search` 进一步检索。")
    with tabs[1]:
        st.dataframe(
            [
                {"字段": "文件名", "值": document.get("filename")},
                {"字段": "来源路径", "值": document.get("source")},
                {"字段": "大小", "值": format_size(document.get("size"))},
                {"字段": "chunk 数", "值": str(document.get("chunks") or 0)},
                {"字段": "页数", "值": str(document.get("pages") or 0)},
                {"字段": "入库时间", "值": document.get("ingested_at") or "-"},
                {"字段": "向量模型", "值": document.get("embedding") or "-"},
                {"字段": "内容哈希", "值": (document.get("sha256") or "-")[:16]},
            ],
            use_container_width=True,
            hide_index=True,
        )
    with tabs[2]:
        st.caption("删除只影响向量库与登记表，不会动原始文件。")
        confirm = st.checkbox(f"确认删除《{document.get('filename')}》", key=f"del-{doc_id}")
        if st.button("删除这份文档", disabled=not confirm, key=f"delbtn-{doc_id}"):
            removed = remove_document(collection, doc_id, config)
            st.success(f"已删除 {removed} 个 chunk")
            st.rerun()


def render_docs(config: Config, collection: str, top_k: int) -> None:
    manager = CollectionManager(config.collections_dir())
    store = manager.get(collection)
    stats = store.stats()
    documents = list_documents(collection, config)
    total_size = sum(int(doc.get("size") or 0) for doc in documents)
    if not total_size:
        total_size = dir_size(store.path)

    columns = st.columns(4)
    with columns[0]:
        st.metric("文档", stats.get("documents", 0))
    with columns[1]:
        st.metric("chunk", stats.get("chunks", 0))
    with columns[2]:
        st.metric("向量维度", stats.get("dim", 0))
    with columns[3]:
        st.metric("占用空间", format_size(total_size))

    if not documents:
        st.info("还没有文档。可以在左侧上传文件，或点击「导入示例文档」。")
    else:
        keyword = st.text_input("按文件名筛选", placeholder="输入文件名片段，例如 实验", key="doc-filter")
        visible = [
            doc for doc in documents if keyword.strip() == "" or keyword.strip() in str(doc.get("filename"))
        ]
        st.dataframe(
            [
                {
                    "文件名": doc.get("filename"),
                    "类型": file_type_label(doc.get("filename")),
                    "大小": format_size(doc.get("size")),
                    "chunk 数": doc.get("chunks"),
                    "页数": doc.get("pages"),
                    "入库时间": doc.get("ingested_at") or "-",
                    "来源路径": short_path(doc.get("source")),
                }
                for doc in visible
            ],
            use_container_width=True,
            hide_index=True,
        )
        if visible:
            selected = st.selectbox(
                "查看文档详情",
                options=[doc.get("filename") for doc in visible],
                index=None,
                placeholder="选择一份文档",
                key="doc-detail",
            )
            if selected:
                document = next(doc for doc in visible if doc.get("filename") == selected)
                render_document_detail(config, collection, document)
        else:
            st.caption("没有匹配的文件名。")

    with st.expander("集合维护：重建索引 / 导出 / 查看体积"):
        st.caption(
            f"数据目录：`{store.path}` ｜ 共 {format_size(dir_size(store.path))}"
        )
        left, right = st.columns(2)
        with left:
            if st.button("重建当前集合索引", use_container_width=True):
                sources = [str(doc.get("source")) for doc in documents if Path(str(doc.get("source"))).is_file()]
                if not sources:
                    st.warning("原始文件已不在原路径，无法重建；请重新上传后再入库。")
                else:
                    st.session_state.last_report = do_ingest(sources, collection, config, force=True)
                    st.rerun()
        with right:
            try:
                payload = export_collection(collection, config)
            except VectorStoreError as exc:
                st.caption(f"暂不可导出：{exc}")
            else:
                st.download_button(
                    "导出当前集合（zip）",
                    data=payload,
                    file_name=f"{collection}.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
        render_report(st.session_state.get("last_report"))

    with st.expander("检索调试：只看召回，不生成回答"):
        query = st.text_input("输入查询文本", key="debug_query")
        if query:
            hits = build_engine(collection, config).retrieve(query, top_k=top_k)
            if not hits:
                st.warning("没有命中任何 chunk，可以调低相关度阈值。")
            for index, hit in enumerate(hits, start=1):
                st.markdown(f"**[{index}] {hit.chunk.citation()}** ｜ score {hit.score:.4f}")
                st.caption(" ".join(hit.chunk.text.split())[:300])


# ------------------------------------------------------------------ 页面：设置
def render_settings(config: Config) -> None:
    st.subheader("当前配置")
    st.caption("配置文件：" + (str(config.source) if config.source else "未找到，使用内置默认值"))

    provider = str(config.get("llm.provider", "ollama"))
    model = str(config.get("llm.model", "qwen2.5:7b"))
    embed_provider = str(config.get("embeddings.provider", "ollama"))
    embed_model = str(config.get("embeddings.model", "nomic-embed-text"))
    rows = [
        ("生成模型", f"{provider} / {model}" if provider != "none" else "未启用（离线提取）"),
        ("向量模型", f"{embed_provider} / {embed_model}"),
        ("切分参数", f"{config.get('chunking.size')} 字符，重叠 {config.get('chunking.overlap')}"),
        ("检索参数", f"top_k={config.get('retrieval.top_k')}，阈值={config.get('retrieval.score_threshold')}"),
        ("数据目录", str(config.collections_dir())),
    ]
    table = ["| 项 | 值 |", "|---|---|"] + [f"| {name} | `{value}` |" for name, value in rows]
    st.markdown("\n".join(table))
    with st.expander("完整配置（JSON）"):
        st.json(config.as_dict(), expanded=True)

    models = cached_models(str(config.get("llm.base_url", "http://127.0.0.1:11434")))
    if models:
        st.subheader("本地已安装模型")
        st.code("\n".join(models))

    st.subheader("数据占用")
    collections_size = dir_size(config.collections_dir())
    conversations_size = dir_size(conversations_dir(config))
    columns = st.columns(3)
    with columns[0]:
        st.metric("向量库", format_size(collections_size))
    with columns[1]:
        st.metric("对话记录", format_size(conversations_size))
    with columns[2]:
        st.metric("上传缓存", format_size(dir_size(config.data_dir() / "uploads")))
    confirm_clear = st.checkbox("确认删除所有集合的对话记录（不影响文档与向量）", key="clear-all-confirm")
    if st.button("清理全部对话记录", disabled=not confirm_clear):
        for path in conversations_dir(config).glob("*.json"):
            path.unlink()
        st.session_state.messages = []
        st.success("已清理所有对话记录")
        st.rerun()

    st.caption(
        "修改配置：编辑项目根目录的 config.yaml 后重启应用；"
        "也可在终端用 `python -m localdocqa check` 检查环境。"
    )


def render_about() -> None:
    st.subheader("这个工具做什么")
    st.markdown(
        "- 把本地文档切分、向量化后存入本机向量库；\n"
        "- 提问时先检索最相关的若干段，再交给本地模型归纳，回答带 `[编号]` 来源；\n"
        "- 文档、向量、对话记录都不离开本机，只有本机模型进程参与推理。"
    )
    st.subheader("v0.2 的体验改进")
    st.markdown(
        "- 空态引导与示例问题，一键发起提问；\n"
        "- 来源按查询词高亮，可直接展开原文并复制；\n"
        "- 对话按集合落盘，刷新或重启后接着聊；可重新生成上一条回答；\n"
        "- 文档详情、逐文件入库报告、集合打包导入导出。"
    )
    st.subheader("目录约定")
    st.code(
        "data/collections/<集合>/chunks.jsonl   文本分块\n"
        "data/collections/<集合>/vectors.npy    向量矩阵\n"
        "data/collections/<集合>/documents.json 文档登记表\n"
        "data/conversations/<集合>.json         对话记录\n"
        "data/uploads/                          网页上传的原始文件",
        language="text",
    )
    st.subheader("技术栈")
    st.caption(
        "Python · Streamlit · numpy · pypdf · python-docx · Ollama ｜ "
        "检索与生成全部在本机完成，无云端调用。"
    )


# ------------------------------------------------------------------ 侧边栏
def render_sidebar(config: Config) -> tuple[str, int, bool]:
    with st.sidebar:
        st.subheader("知识库")
        collections = collection_options()
        current = st.session_state.get("collection", collections[0])
        if current not in collections:
            current = collections[0]
        collection = st.selectbox("当前集合", collections, index=collections.index(current))
        st.session_state.collection = collection

        with st.expander("管理集合"):
            new_name = st.text_input("新集合名称", placeholder="例如：毕业论文", key="new-collection")
            if st.button("创建集合", use_container_width=True):
                if new_name.strip():
                    manager = CollectionManager(config.collections_dir())
                    manager.ensure(new_name.strip())
                    st.session_state.collection = CollectionManager.normalize_name(new_name)
                    st.success(f"已创建集合：{st.session_state.collection}")
                    st.rerun()
                else:
                    st.warning("请输入集合名称")

            archive_file = st.file_uploader(
                "导入集合（.zip）",
                type=["zip"],
                key="import-archive",
                help="由「文档」页导出的集合压缩包",
            )
            if st.button("导入集合", disabled=archive_file is None, use_container_width=True):
                try:
                    imported = import_collection(archive_file.getvalue(), config)
                except VectorStoreError as exc:
                    st.error(f"导入失败：{exc}")
                else:
                    st.session_state.collection = imported
                    st.success(f"已导入集合：{imported}")
                    st.rerun()

            confirm = st.checkbox("我确认要删除当前集合的全部数据", key="delete-collection")
            if st.button("删除当前集合", disabled=not confirm, use_container_width=True):
                CollectionManager(config.collections_dir()).delete(collection)
                clear_conversation(collection, config)
                st.session_state.messages = []
                st.session_state.pop("collection", None)
                st.session_state.loaded_collection = None
                st.warning(f"已删除集合：{collection}")
                st.rerun()

        st.divider()
        st.subheader("导入文档")
        uploaded = st.file_uploader(
            "选择文件（可多选）",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            help="支持 PDF / Word / Markdown / TXT / CSV / JSON / HTML",
            key="uploader",
        )
        force = st.checkbox("忽略重复，强制重新入库")
        if st.button("开始入库", type="primary", use_container_width=True, disabled=not uploaded):
            upload_dir = config.data_dir() / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            saved: list[str] = []
            for item in uploaded:
                target = upload_dir / Path(item.name).name
                target.write_bytes(item.getbuffer())
                saved.append(str(target))
            st.session_state.last_report = do_ingest(saved, collection, config, force=force)
            st.rerun()

        if SAMPLE_DIR.is_dir() and st.button("导入示例文档", use_container_width=True):
            st.session_state.last_report = do_ingest([str(SAMPLE_DIR)], collection, config, force=force)
            st.rerun()

        render_report(st.session_state.get("last_report"), details=False)

        st.divider()
        with st.expander("检索参数", expanded=True):
            top_k = st.slider("召回条数 top_k", 1, 12, int(config.get("retrieval.top_k", 5)))
            threshold = st.slider(
                "相关度阈值", -1.0, 1.0, float(config.get("retrieval.score_threshold", 0.25)), 0.01
            )
            stream_mode = st.toggle("流式输出", value=True)

        config.set("retrieval.top_k", top_k)
        config.set("retrieval.score_threshold", threshold)

        st.divider()
        st.subheader("运行状态")
        try:
            embedder = cached_embeddings(json.dumps(config.section("embeddings"), sort_keys=True))
            st.caption(f"向量后端：`{embedder.name}`")
        except EmbeddingError as exc:
            st.caption(f"向量后端异常：{exc}")

        provider = str(config.get("llm.provider", "ollama"))
        base_url = str(config.get("llm.base_url", "http://127.0.0.1:11434"))
        model = str(config.get("llm.model", "qwen2.5:7b"))
        healthy, message = cached_llm_status(provider, base_url, model)
        if provider in ("none", "off", "extractive", ""):
            st.caption("模型：未启用（离线提取模式）")
        else:
            st.caption(f"模型：`{model}` {'可用' if healthy else '不可用'}")
            if not healthy:
                st.caption(message)
                with st.expander("Ollama 安装提示"):
                    st.code(f"ollama serve\nollama pull {model}\nollama pull nomic-embed-text")

    return collection, top_k, stream_mode


# ------------------------------------------------------------------ 入口
def main() -> None:
    config = Config.load()
    st.set_page_config(page_title="LocalDocQA 本地文档问答", page_icon="📄", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_report", None)
    st.session_state.setdefault("loaded_collection", None)

    collection, top_k, stream_mode = render_sidebar(config)
    if st.session_state.get("loaded_collection") != collection:
        st.session_state.messages = load_conversation(collection, config)
        st.session_state.loaded_collection = collection

    page = st.radio("功能", PAGES, horizontal=True, label_visibility="collapsed")
    st.title("LocalDocQA · 本地文档问答助手")
    st.caption("文档与向量全部保存在本机，只在本地调用模型，不上传任何内容到云端。")

    if page == "问答":
        render_chat(config, collection, top_k, stream_mode)
    elif page == "文档":
        render_docs(config, collection, top_k)
    elif page == "设置":
        render_settings(config)
    else:
        render_about()


if __name__ == "__main__":
    main()
