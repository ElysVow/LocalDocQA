"""Streamlit 网页界面：上传入库、对话问答、来源溯源、文档管理。

启动：
    streamlit run app.py

布局说明：页面切换用侧边栏单选，而不是 st.tabs —— st.chat_input 只有在主区域
（不在 tab/container 内）才会固定在视口底部，放进 tab 会变成行内元素。
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

from localdocqa.config import Config  # noqa: E402
from localdocqa.embeddings import EmbeddingError, build_embeddings, list_ollama_models  # noqa: E402
from localdocqa.ingest import IngestReport, ingest, list_documents, remove_document  # noqa: E402
from localdocqa.llm import build_llm  # noqa: E402
from localdocqa.rag_chain import RAGEngine  # noqa: E402
from localdocqa.vector_store import CollectionManager  # noqa: E402

SAMPLE_DIR = ROOT / "examples" / "sample_docs"
UPLOAD_TYPES = ["pdf", "docx", "md", "markdown", "txt", "log", "csv", "json", "html", "htm"]
PAGES = ["问答", "文档", "设置", "关于"]


def short_path(source: object) -> str:
    """表格里显示相对路径，绝对路径太长会把列撑爆。"""
    if not source:
        return "-"
    path = Path(str(source))
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path.name


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


# ------------------------------------------------------------------ 工具
def collection_options() -> list[str]:
    existing = CollectionManager(Config.load().collections_dir()).list_collections()
    return existing or ["default"]


def build_engine(collection: str, config: Config) -> RAGEngine:
    signature = json.dumps(config.section("embeddings"), sort_keys=True)
    return RAGEngine(config, collection=collection, embeddings=cached_embeddings(signature))


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"参考来源（{len(sources)} 段）", expanded=False):
        for source in sources:
            st.markdown(
                f"**[{source['index']}] {source['citation']}**  \n"
                f"<span style='color:#8a8a8a;font-size:12px'>"
                f"相似度 {source['score']} ｜ chunk #{source['chunk_index']}</span>",
                unsafe_allow_html=True,
            )
            preview = html.escape(source["preview"])
            st.markdown(
                "<div style='font-size:12.5px;line-height:1.65;color:#555;background:#fafafa;"
                "border-left:3px solid #d8dde5;padding:8px 10px;margin:4px 0 12px 0;'>"
                f"{preview}</div>",
                unsafe_allow_html=True,
            )


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


def do_ingest(paths: list[str], collection: str, config: Config, force: bool = False) -> IngestReport:
    progress_bar = st.progress(0.0, text="准备中…")

    def on_progress(message: str, fraction: float | None) -> None:
        progress_bar.progress(min(max(fraction or 0.0, 0.0), 1.0), text=message)

    report = ingest(paths, collection=collection, config=config, force=force, progress=on_progress)
    progress_bar.empty()
    return report


def render_report(report: IngestReport | None) -> None:
    if not report:
        return
    if report.errors:
        st.warning(report.summary())
        for error in report.errors:
            st.caption(f"{Path(error['file']).name}：{error['error']}")
    else:
        st.success(report.summary())


# ------------------------------------------------------------------ 页面：问答
def render_chat(config: Config, collection: str, top_k: int, stream_mode: bool) -> None:
    left, right = st.columns([3, 1])
    with left:
        stats = CollectionManager(config.collections_dir()).get(collection).stats()
        st.caption(
            f"当前集合 **{collection}**：{stats.get('documents', 0)} 个文档 / "
            f"{stats.get('chunks', 0)} 个 chunk"
        )
    with right:
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    engine = build_engine(collection, config)

    if st.session_state.messages:
        st.download_button(
            "导出对话（Markdown）",
            data=messages_to_markdown(st.session_state.messages, collection),
            file_name=f"{collection}-qa-{datetime.now().strftime('%Y%m%d-%H%M')}.md",
            mime="text/markdown",
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    prompt = st.chat_input("针对文档提问，例如：这份文档的核心结论是什么？")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
    with st.chat_message("assistant"):
        if stream_mode:
            placeholder = st.empty()
            placeholder.markdown("*正在检索文档…*")
            buffer: list[str] = []
            sources: list[dict] = []
            meta = {"used_llm": False, "elapsed": 0.0}
            for event in engine.stream(prompt, history=history):
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
                elif event["type"] == "error":
                    placeholder.error(event["message"])
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"出错了：{event['message']}"}
                    )
                    st.stop()

            st.caption(
                f"耗时 {meta['elapsed']:.1f}s ｜ "
                f"{'模型生成' if meta['used_llm'] else '离线提取（未接入大模型）'}"
            )
            render_sources(sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": "".join(buffer), "sources": sources}
            )
        else:
            answer = engine.answer(prompt, history=history)
            st.markdown(answer.text)
            st.caption(
                f"耗时 {answer.elapsed:.1f}s ｜ "
                f"{'模型生成' if answer.used_llm else '离线提取（未接入大模型）'}"
            )
            render_sources(answer.sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer.text, "sources": answer.sources}
            )


# ------------------------------------------------------------------ 页面：文档
def render_docs(config: Config, collection: str, top_k: int) -> None:
    st.subheader(f"集合「{collection}」的文档")
    documents = list_documents(collection, config)
    if documents:
        st.dataframe(
            [
                {
                    "文件名": doc.get("filename"),
                    "chunk 数": doc.get("chunks"),
                    "页数": doc.get("pages"),
                    "入库时间": doc.get("ingested_at"),
                    "来源路径": short_path(doc.get("source")),
                }
                for doc in documents
            ],
            use_container_width=True,
            hide_index=True,
        )
        target = st.selectbox(
            "选择要删除的文档",
            options=[doc.get("filename") for doc in documents],
            index=None,
            placeholder="选择文件名",
        )
        if target:
            doc_id = next(d["doc_id"] for d in documents if d.get("filename") == target)
            if st.button(f"删除《{target}》"):
                removed = remove_document(collection, str(doc_id), config)
                st.success(f"已删除 {removed} 个 chunk")
                st.rerun()
    else:
        st.info("还没有文档。可以在左侧上传文件，或点击「导入示例文档」。")

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
    st.subheader("目录约定")
    st.code(
        "data/collections/<集合>/chunks.jsonl   文本分块\n"
        "data/collections/<集合>/vectors.npy    向量矩阵\n"
        "data/collections/<集合>/documents.json 文档登记表\n"
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

        with st.expander("新建 / 删除集合"):
            new_name = st.text_input("新集合名称", placeholder="例如：毕业论文")
            if st.button("创建集合", use_container_width=True):
                if new_name.strip():
                    CollectionManager(config.collections_dir()).ensure(new_name.strip())
                    st.session_state.collection = CollectionManager.normalize_name(new_name)
                    st.success(f"已创建集合：{st.session_state.collection}")
                    st.rerun()
                else:
                    st.warning("请输入集合名称")

            confirm = st.checkbox("我确认要删除当前集合的全部数据")
            if st.button("删除当前集合", disabled=not confirm, use_container_width=True):
                CollectionManager(config.collections_dir()).delete(collection)
                st.session_state.messages = []
                st.session_state.pop("collection", None)
                st.warning(f"已删除集合：{collection}")
                st.rerun()

        st.divider()
        st.subheader("导入文档")
        uploaded = st.file_uploader(
            "选择文件（可多选）",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            help="支持 PDF / Word / Markdown / TXT / CSV / JSON / HTML",
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

        render_report(st.session_state.get("last_report"))

        st.divider()
        st.subheader("检索参数")
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
    st.set_page_config(page_title="LocalDocQA 本地文档问答", layout="wide")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_report", None)

    collection, top_k, stream_mode = render_sidebar(config)

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
