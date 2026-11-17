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

import html
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
from localdocqa.embeddings import build_embeddings, list_ollama_models  # noqa: E402
from localdocqa.ingest import IngestReport, ingest, list_documents, remove_document  # noqa: E402
from localdocqa.rag_chain import RAGEngine  # noqa: E402
from localdocqa.text_utils import clean_text, highlight  # noqa: E402
from localdocqa.vector_store import CollectionManager, VectorStoreError  # noqa: E402

SAMPLE_DIR = ROOT / "examples" / "sample_docs"
UPLOAD_TYPES = ["pdf", "docx", "md", "markdown", "txt", "log", "csv", "json", "html", "htm"]
PAGES = ["问答", "文档", "设置", "关于"]

CSS = """
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 7rem; max-width: 1060px;}
h1 {font-size: 1.55rem !important; font-weight: 650 !important; letter-spacing: -0.02em; margin-bottom: 0.2rem;}
h2 {font-size: 1.2rem !important; margin-top: 1.5rem;}
h3 {font-size: 0.98rem !important; letter-spacing: 0.01em;}
[data-testid="stMarkdownContainer"] p {line-height: 1.72;}
[data-testid="stCaptionContainer"] {color: #868686;}
[data-testid="stSidebar"] {background: #fbfbfb;}
[data-testid="stSidebar"] h3 {font-size: 0.9rem !important; color: #4c4c4c;}
[data-testid="stChatMessage"] {padding: 0.3rem 0;}
[data-testid="stChatInput"] textarea {font-size: 0.94rem;}
mark {background: #ffe58f; color: inherit; padding: 0 2px; border-radius: 3px;}
.stButton > button, div[data-testid="stDownloadButton"] > button {border-radius: 8px;}
hr {margin: 1rem 0;}
/* 展开面板去掉外框与底色，只保留一行标题，减少页面上的盒子 */
[data-testid="stExpander"], [data-testid="stExpander"] details,
[data-testid="stExpander"] summary {background: transparent !important; border: none !important; box-shadow: none !important;}
[data-testid="stExpander"] summary {padding: 0.25rem 0 !important; font-size: 0.87rem !important; color: #6f6f6f !important;}
[data-testid="stExpander"] summary:hover {color: #222 !important;}
/* 顶部导航改成纯文字标签：隐藏圆点，当前页用下划线标示 */
[data-testid="stRadioGroup"] {gap: 1.8rem !important;}
[data-testid="stRadioOption"] {padding: 0 !important;}
[data-testid="stRadioOption"] > div > div:first-child {display: none !important;}
[data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p {font-size: 0.9rem; color: #777;}
[data-testid="stRadioOption"][data-selected="true"] [data-testid="stMarkdownContainer"] p {
    color: #1f1f1f; font-weight: 600; border-bottom: 2px solid #1f1f1f; padding-bottom: 2px;}
/* 来源段落：细线分隔，不再一张卡片套一张卡片 */
.src {padding: 0.55rem 0 0.8rem; border-bottom: 1px solid #f0f0f0;}
.src:last-child {border-bottom: none;}
.src-head {font-size: 0.9rem; font-weight: 600; color: #333;}
.src-meta {font-size: 0.76rem; color: #949494; margin: 0.15rem 0 0.4rem;}
.src-text {font-size: 0.84rem; line-height: 1.7; color: #5a5a5a;
          border-left: 2px solid #e2e5ea; padding-left: 0.65rem;}
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
    with st.expander(f"来源 · {len(sources)} 段", expanded=False):
        for source in sources:
            body = highlight(source.get("preview") or "", query)
            st.markdown(
                f"<div class='src'>"
                f"<div class='src-head'>[{source['index']}] {html.escape(str(source['citation']))}</div>"
                f"<div class='src-meta'>相似度 {source['score']} · 第 {int(source['chunk_index']) + 1} 段</div>"
                f"<div class='src-text'>{body}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            full_text = source.get("text") or source.get("preview") or ""
            with st.popover("原文", type="tertiary"):
                st.code(full_text, language=None)


def split_offline_answer(text: str) -> tuple[str, bool]:
    """离线摘录的固定前缀和结尾提示不进入正文，只在下面留一行小字说明。"""
    marker = "当前为**离线提取模式**"
    if not text.startswith(marker):
        return text, False
    body = text.split("\n", 1)[1] if "\n" in text else ""
    body = body.split("\n\n> 本地模型未连接", 1)[0]
    return body.strip(), True


def render_answer(text: str) -> None:
    body, offline = split_offline_answer(text)
    st.markdown(body)
    meta_col, copy_col = st.columns([8, 1], vertical_alignment="center")
    with meta_col:
        if offline:
            st.caption("离线摘录 · 未连接生成模型")
    with copy_col:
        with st.popover("复制", type="tertiary", use_container_width=True):
            st.code(text, language=None)


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


def render_start(collection: str, config: Config) -> None:
    """空对话时的轻量入口：一句说明 + 三个示例，不占版面。"""
    st.caption("资料会先在本机建索引，回答会标注原文来源。可以直接输入，也可以从示例开始：")
    questions = example_questions(collection, config)
    for column, question in zip(st.columns(len(questions)), questions, strict=False):
        with column:
            if st.button(question, key=f"example-{question}", type="tertiary", use_container_width=True):
                st.session_state.pending_prompt = question
                st.rerun()


def render_chat(config: Config, collection: str, top_k: int, stream_mode: bool) -> None:
    stats = CollectionManager(config.collections_dir()).get(collection).stats()
    left, actions = st.columns([6, 1], vertical_alignment="center")
    with left:
        st.caption(
            f"{collection} · {stats.get('documents', 0)} 份文档 · {stats.get('chunks', 0)} 个片段"
        )
    with actions:
        if st.session_state.messages:
            with st.popover("对话", type="tertiary", use_container_width=True):
                st.download_button(
                    "导出 Markdown",
                    data=messages_to_markdown(st.session_state.messages, collection),
                    file_name=f"{collection}-qa-{datetime.now().strftime('%Y%m%d-%H%M')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
                if (
                    st.session_state.messages[-1]["role"] == "assistant"
                    and len(st.session_state.messages) >= 2
                ):
                    if st.button("重新生成上一条", key="regenerate", use_container_width=True):
                        last_question = st.session_state.messages[-2]["content"]
                        st.session_state.messages.pop()
                        if st.session_state.messages[-1]["role"] == "user":
                            st.session_state.messages.pop()
                        save_conversation(collection, st.session_state.messages, config)
                        st.session_state.pending_prompt = last_question
                        st.rerun()
                st.caption("清空后无法恢复")
                confirm_clear = st.checkbox("确认清空对话", key="clear-chat-confirm")
                if st.button("清空对话", disabled=not confirm_clear, use_container_width=True):
                    st.session_state.messages = []
                    clear_conversation(collection, config)
                    st.rerun()

    intro = st.empty()
    if not st.session_state.messages:
        with intro.container():
            render_start(collection, config)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer(message["content"])
            else:
                st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"], message.get("query", ""))

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
                for event in engine.stream(request, history=history):
                    if event["type"] == "token":
                        buffer.append(event["text"])
                        placeholder.markdown("".join(buffer) + "▌")
                    elif event["type"] == "sources":
                        sources = event["sources"]
                    elif event["type"] == "done":
                        answer = event["answer"]
                        sources = answer.sources
                        body, _ = split_offline_answer(answer.text)
                        placeholder.markdown(body)
                        buffer = [answer.text]
                    elif event["type"] == "error":
                        placeholder.error(event["message"])
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"出错了：{event['message']}"}
                        )
                        save_conversation(collection, st.session_state.messages, config)
                        st.stop()
                answer_text = "".join(buffer)
                render_sources(sources, request)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer_text, "sources": sources, "query": request}
                )
            else:
                answer = engine.answer(request, history=history)
                render_answer(answer.text)
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
        st.caption(f"共 {len(chunks)} 段")
        for chunk in chunks[:6]:
            page = f" · 第 {chunk.page} 页" if chunk.page else ""
            st.caption(f"第 {chunk.chunk_index + 1} 段{page}")
            text = clean_text(chunk.text)
            st.markdown(text[:900] + ("…" if len(text) > 900 else ""))
            st.divider()
        if len(chunks) > 6:
            st.caption(f"还有 {len(chunks) - 6} 段未展示。")
    with tabs[1]:
        st.dataframe(
            [
                {"字段": "文件名", "值": document.get("filename")},
                {"字段": "来源路径", "值": document.get("source")},
                {"字段": "大小", "值": format_size(document.get("size"))},
                {"字段": "片段数", "值": str(document.get("chunks") or 0)},
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

    st.caption(
        f"{stats.get('documents', 0)} 份文档 · {stats.get('chunks', 0)} 个片段 · "
        f"{format_size(total_size)}"
    )
    if not documents:
        st.info("还没有文档。可以在左侧「导入文档」里上传，或先载入示例。")
    else:
        keyword = st.text_input(
            "筛选", placeholder="按文件名筛选", label_visibility="collapsed", key="doc-filter"
        )
        visible = [
            doc for doc in documents if keyword.strip() == "" or keyword.strip() in str(doc.get("filename"))
        ]
        st.dataframe(
            [
                {
                    "文件名": doc.get("filename"),
                    "类型": file_type_label(doc.get("filename")),
                    "大小": format_size(doc.get("size")),
                    "片段数": doc.get("chunks"),
                    "页数": str(doc.get("pages")) if doc.get("pages") else "-",
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
                placeholder="打开一份文档查看详情",
                label_visibility="collapsed",
                key="doc-detail",
            )
            if selected:
                document = next(doc for doc in visible if doc.get("filename") == selected)
                render_document_detail(config, collection, document)
        else:
            st.caption("没有匹配的文件名。")

    with st.expander("集合维护"):
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
                    "导出集合 zip",
                    data=payload,
                    file_name=f"{collection}.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
        render_report(st.session_state.get("last_report"))

    with st.expander("检索调试"):
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
    st.caption("配置文件：" + (short_path(config.source) if config.source else "使用内置默认值"))

    provider = str(config.get("llm.provider", "ollama"))
    model = str(config.get("llm.model", "qwen2.5:7b"))
    embed_provider = str(config.get("embeddings.provider", "ollama"))
    embed_model = str(config.get("embeddings.model", "nomic-embed-text"))
    rows = [
        ("生成模型", f"{provider} / {model}" if provider != "none" else "未启用（离线提取）"),
        ("向量模型", f"{embed_provider} / {embed_model}"),
        ("切分", f"{config.get('chunking.size')} 字 / 重叠 {config.get('chunking.overlap')} 字"),
        ("检索", f"召回 {config.get('retrieval.top_k')} 段 · 阈值 {config.get('retrieval.score_threshold')}"),
        ("数据目录", short_path(config.collections_dir())),
    ]
    table = ["| 项 | 值 |", "|---|---|"] + [f"| {name} | `{value}` |" for name, value in rows]
    st.markdown("\n".join(table))
    with st.expander("完整配置（JSON）"):
        st.json(config.as_dict(), expanded=True)

    models = cached_models(str(config.get("llm.base_url", "http://127.0.0.1:11434")))
    if models:
        st.subheader("本地已安装模型")
        st.code("\n".join(models))

    collections_size = dir_size(config.collections_dir())
    conversations_size = dir_size(conversations_dir(config))
    st.caption(
        f"向量库 {format_size(collections_size)} · 对话 {format_size(conversations_size)} · "
        f"上传缓存 {format_size(dir_size(config.data_dir() / 'uploads'))}"
    )
    with st.expander("清理对话记录"):
        st.caption("只删除对话 JSON，不影响文档、向量与原始文件。")
        confirm_clear = st.checkbox("确认删除所有集合的对话记录", key="clear-all-confirm")
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
        collections = collection_options()
        current = st.session_state.get("collection", collections[0])
        if current not in collections:
            current = collections[0]
        collection = st.selectbox(
            "知识库", collections, index=collections.index(current), label_visibility="visible"
        )
        st.session_state.collection = collection
        store = CollectionManager(config.collections_dir()).get(collection)
        document_count = store.stats().get("documents", 0)

        with st.expander("导入文档", expanded=document_count == 0):
            uploaded = st.file_uploader(
                "选择文件（可多选）",
                type=UPLOAD_TYPES,
                accept_multiple_files=True,
                help="PDF / Word / Markdown / TXT / CSV / JSON / HTML",
                key="uploader",
            )
            if uploaded:
                if st.button(f"入库 {len(uploaded)} 个文件", type="primary", use_container_width=True):
                    upload_dir = config.data_dir() / "uploads"
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    saved: list[str] = []
                    for item in uploaded:
                        target = upload_dir / Path(item.name).name
                        target.write_bytes(item.getbuffer())
                        saved.append(str(target))
                    st.session_state.last_report = do_ingest(saved, collection, config)
                    st.rerun()
            if document_count == 0 and SAMPLE_DIR.is_dir():
                if st.button("先导入 3 份示例文档", type="tertiary", use_container_width=True):
                    st.session_state.last_report = do_ingest([str(SAMPLE_DIR)], collection, config)
                    st.rerun()

        render_report(st.session_state.get("last_report"), details=False)

        with st.expander("集合管理"):
            st.caption("新建集合")
            new_name = st.text_input(
                "新建集合",
                placeholder="例如：毕业论文",
                label_visibility="collapsed",
                key="new-collection",
            )
            if st.button("创建", use_container_width=True):
                if new_name.strip():
                    manager = CollectionManager(config.collections_dir())
                    manager.ensure(new_name.strip())
                    st.session_state.collection = CollectionManager.normalize_name(new_name)
                    st.rerun()
                else:
                    st.warning("请输入集合名称")

            st.caption("导入集合 zip")
            archive_file = st.file_uploader(
                "导入集合",
                type=["zip"],
                key="import-archive",
                label_visibility="collapsed",
            )
            if st.button("导入", disabled=archive_file is None, use_container_width=True):
                try:
                    imported = import_collection(archive_file.getvalue(), config)
                except VectorStoreError as exc:
                    st.error(f"导入失败：{exc}")
                else:
                    st.session_state.collection = imported
                    st.rerun()

            st.caption("删除集合")
            confirm = st.checkbox("确认删除当前集合及对话", key="delete-collection")
            if st.button("删除", disabled=not confirm, use_container_width=True):
                CollectionManager(config.collections_dir()).delete(collection)
                clear_conversation(collection, config)
                st.session_state.messages = []
                st.session_state.pop("collection", None)
                st.session_state.loaded_collection = None
                st.rerun()

    top_k = int(config.get("retrieval.top_k", 5))
    return collection, top_k, True


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
    st.title("LocalDocQA" if page == "问答" else page)

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
