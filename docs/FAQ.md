# 常见问题

## 安装与启动

**Q：没有独立显卡，能跑吗？**
能。`qwen2.5:3b` 甚至 `qwen2.5:1.5b` 在 CPU 上也能出结果，只是慢（7B 在纯 CPU 上单次回答常见 10~30 秒）。
把 `config.yaml` 的 `llm.model` 改小即可。向量模型 `nomic-embed-text` 很小，CPU 上入库几百页没问题。

**Q：完全不想装 Ollama，能体验吗？**
能。`config.yaml` 里设 `llm.provider: none` 和 `embeddings.provider: hash`，
工具会走"离线提取模式"：检索照常工作，回答是原文摘录而不是归纳。这也是 CI 里跑的模式。

**Q：`streamlit run app.py` 报 `ModuleNotFoundError: localdocqa`**
没有安装本包。执行 `pip install -e ".[ui]"`；或者临时用 `$env:PYTHONPATH="src"`（Windows）/
`export PYTHONPATH=src`（macOS/Linux）再启动。

## 入库

**Q：导入 PDF 提示"未提取到文本"**
该 PDF 是扫描件（图片页，没有文字层）。当前版本不做 OCR，需要先用其他工具转成文字版 PDF；
OCR 支持在路线图 v0.2 里。

**Q：同一个文件改了内容，重新导入会怎样？**
内容变了 SHA256 就变了，会作为新文档入库；旧版本仍留在库里。请先在"文档"页删除旧版本再导入，
或入库时加 `--force`（会先移除同 SHA 的旧数据，但改名后的旧内容仍会残留）。

**Q：重复导入会不会撑爆向量库？**
不会。去重口径是文件内容哈希，同样的文件导入第二次会全部计入"跳过重复"。

**Q：支持多大的单文件？**
没有硬限制，但一份 PDF 超过 200 页时建议拆分，因为 `llm.max_context_chars` 只装得下少量片段，
再长的文档对单次问答没有额外收益。

**Q：CSV / JSON 也能入库吗？**
能，CSV 每行转成"列名: 值"的句子，JSON 会摊平成 `a.b.c: 值` 的形式。它们适合做参数表、配置说明类问答。

## 检索与回答

**Q：回答里说"资料中没有相关内容"，但我知道文档里有**
三种常见原因：一是问法用词和原文差异太大（离线词法模式尤其明显）；二是相关段落被切分切断了，
把 `retrieval.expand_neighbors` 调成 2；三是阈值过高，把 `retrieval.score_threshold` 从 0.25 降到 0.15 试试。
排查方法：用 `python -m localdocqa search "你的问题" -c 集合名` 直接看召回结果和分数。

**Q：回答里的 `[2]` 找不到对应的来源？**
这属于 bug。编号由 `format_context()` 统一生成，界面和 prompt 用同一份数据；
如果确实对不上，请附上问题、集合名和 `data/collections/<集合>/chunks.jsonl` 里的相关行提 issue。

**Q：能不能让它回答文档之外的问题？**
设计上不允许。system prompt 明确要求"只使用参考资料"，资料不足时直说。
这是为了避免把模型幻觉包装成"文档问答"——这类工具最怕的不是答不出，而是答得很像真的。

**Q：换 embedding 模型后报"向量维度不一致"**
向量库维度写死在 `meta.json` 里。换模型（例如从 256 维哈希换成 768 维 nomic）必须新建集合重新入库。
旧集合可以保留，两者互不影响。

**Q：`score_threshold` 和 `lexical_relative_cutoff` 是什么关系？**
前者只作用于语义向量（绝对余弦阈值）；后者只作用于离线词法向量（相对最高分的比例）。
两套阈值不混用，因为两种向量的分数分布不在一个量级。

## 数据与隐私

**Q：真的不会联网吗？**
除首次 `pip install` 和 `ollama pull` 下载模型外，运行时不发起网络请求。所有推理都打向
`llm.base_url`（默认 `127.0.0.1:11434`，即本机 Ollama）。如果你把 `base_url` 改成远程地址，
数据当然会发到那台机器——这一点由配置决定，不是工具偷偷做的。

**Q：数据存在哪，怎么备份？**
`data/collections/<集合>/` 整个目录拷走即可，三个文件就是全部索引；`data/uploads/` 是网页上传的原始文件。
直接打包 `data/` 就能迁移到另一台机器（embedding 模型相同则无需重建）。

**Q：删库会不会误删我的原始文档？**
`data/collections` 里只有解析后的文本块和向量，原始文件在 `data/uploads/` 或你导入时指定的路径。
删除集合只影响索引，不会动原始文件。

## 开发

**Q：为什么不用 LangChain？**
一条链路里用了它三个组件，却要引入几十个传递依赖，版本冲突排查成本比省下的代码高。
这里的切分、检索、prompt 拼装一共几百行，逻辑可见、可测、可改。要用 LangChain 的人也可以把
`RAGEngine` 换掉，底层 `loader` / `vector_store` 仍然能复用。

**Q：为什么自建向量库而不是 Chroma？**
依赖更少（只要 numpy），格式透明（jsonl + npy 能直接打开看），十万条以内延迟毫秒级。
规模上去后换 FAISS 只改 `VectorStore` 一个类，`search()` 签名不变。

**Q：提交 PR 前要跑什么？**
`ruff check .`、`pytest`、`python scripts/smoke_test.py` 三件事都过一遍，见 [CONTRIBUTING.md](../CONTRIBUTING.md)。
