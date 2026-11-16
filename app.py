"""Streamlit 网页界面：上传入库、对话问答、来源溯源、文档管理。

启动：
    streamlit run app.py
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

from localdocqa.config import Config  # noqa: E402
from localdocqa.embeddings import EmbeddingError, build_embeddings, list_ollama_models  # noqa: E402
from localdocqa.ingest import IngestReport, ingest, list_documents, remove_document  # noqa: E402
from localdocqa.llm import build_llm  # noqa: E402
from localdocqa.rag_chain import RAGEngine  # noqa: E402
from localdocqa.vector_store import CollectionManager  # noqa: E402

SAMPLE_DIR = ROOT / "examples" / "sample_docs"
UPLOAD_TYPES = ["pdf", "docx", "md", "markdown", "txt", "log", "csv", "json", "html", "htm"]


# ------------------------------------------------------------------ 缓存
@st.cache_resource(show_spinner=False)
def cached_embeddings(signature: str):
    """按配置签名缓存 embedding 客户端，避免每次交互都去探测模型。"""
    config = Config.load()
    return build_embeddings(config.section("embeddings"))


@st.cache_data(show_spinner=False, ttl=300)
def cached_models(base_url: str):
    return list_ollama_models(base_url)


# ------------------------------------------------------------------ 工具
def collection_options() -> list[str]:
    manager = CollectionManager(Config.load().collections_dir())
    existing = manager.list_collections()
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
            st.caption(source["preview"])


def messages_to_markdown(messages: list[dict], collection: str) -> str:
    lines = [
        f"# {collection} 问答记录",
        "",
        f"导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for message in messages:
        role = "我" if message["role"] == "user" else "LocalDocQA"
        lines.append(f"## {role}")
        lines.append("")
        lines.append(message["content"])
        sources = message.get("sources") or []
        if sources:
            lines.append("")
            lines.append("**来源**")
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


# ------------------------------------------------------------------ 页面
def main() -> None:
    config = Config.load()
    st.set_page_config(page_title="LocalDocQA 本地文档问答", layout="wide")
    st.title("LocalDocQA · 本地文档问答助手")
    st.caption("文档与向量全部保存在本机，只在本地调用模型，不上传任何内容到云端。")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_report" not in st.session_state:
        st.session_state.last_report = None

    # -------------------------------------------------------------- 侧边栏
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
                    manager = CollectionManager(config.collections_dir())
                    manager.ensure(new_name.strip())
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

        if SAMPLE_DIR.is_dir() and st.button("导入示例文档", use_container_width=True):
            st.session_state.last_report = do_ingest([str(SAMPLE_DIR)], collection, config, force=force)

        report = st.session_state.last_report
        if report:
            if report.errors:
                st.warning(report.summary())
                for error in report.errors:
                    st.caption(f"{Path(error['file']).name}：{error['error']}")
            else:
                st.success(report.summary())

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

        client = build_llm(config.section("llm"))
        if client is None:
            st.caption("模型：未启用（离线提取模式）")
        else:
            healthy, message = client.health_check()
            st.caption(f"模型：`{client.name}` {'可用' if healthy else '不可用'}")
            if not healthy:
                st.caption(message)
                with st.expander("Ollama 安装提示"):
                    st.code("ollama serve\nollama pull qwen2.5:7b\nollama pull nomic-embed-text")

    # -------------------------------------------------------------- 主区域
    tab_chat, tab_docs, tab_settings, tab_about = st.tabs(["问答", "文档", "设置", "关于"])

    with tab_chat:
        header_left, header_right = st.columns([3, 1])
        with header_left:
            stats = CollectionManager(config.collections_dir()).get(collection).stats()
            st.caption(
                f"当前集合 **{collection}**：{stats.get('documents', 0)} 个文档 / "
                f"{stats.get('chunks', 0)} 个 chunk"
            )
        with header_right:
            if st.button("清空对话", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        engine = build_engine(collection, config)

        if st.session_state.messages:
            export = messages_to_markdown(st.session_state.messages, collection)
            st.download_button(
                "导出对话（Markdown）",
                data=export,
                file_name=f"{collection}-qa-{datetime.now().strftime('%Y%m%d-%H%M')}.md",
                mime="text/markdown",
            )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    render_sources(message["sources"])

        prompt = st.chat_input("针对文档提问，例如：这份文档的核心结论是什么？")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            with st.chat_message("assistant"):
                if stream_mode:
                    placeholder = st.empty()
                    buffer: list[str] = []
                    sources: list[dict] = []
                    meta: dict = {}
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

                    footer = (
                        f"耗时 {meta.get('elapsed', 0):.1f}s ｜ "
                        f"{'模型生成' if meta.get('used_llm') else '离线提取（未接入大模型）'}"
                    )
                    st.caption(footer)
                    render_sources(sources)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "".join(buffer),
                            "sources": sources,
                        }
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

    with tab_docs:
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
                        "路径": doc.get("source"),
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
                if st.button(f"删除《{target}》", type="secondary"):
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

    with tab_settings:
        st.subheader("当前配置")
        st.caption("配置文件：" + (str(config.source) if config.source else "未找到，使用内置默认值"))
        st.json(config.as_dict(), expanded=False)

        models = cached_models(str(config.get("llm.base_url", "http://127.0.0.1:11434")))
        if models:
            st.subheader("本地已安装模型")
            st.code("\n".join(models))
        st.caption(
            "修改配置：编辑项目根目录的 config.yaml 后重启应用；"
            "也可在终端用 `python -m localdocqa check` 检查环境。"
        )

    with tab_about:
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
        st.caption("图片来源/图标未内置，如需截图可自行补充到 assets/。")


if __name__ == "__main__":
    main()
