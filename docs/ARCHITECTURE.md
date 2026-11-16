# 架构说明

本文说明 LocalDocQA 各模块的职责边界、数据如何流动、以及为什么这样拆分。想改代码前先读这一页，
比直接翻源码省时间。

## 模块职责

| 模块 | 职责 | 不负责 |
| --- | --- | --- |
| `config.py` | 默认值、`config.yaml` 深合并、环境变量覆盖、路径解析 | 校验业务参数合法性以外的逻辑 |
| `loader.py` | 按后缀分发解析器、编码识别、目录扫描、SHA256 | 切分、去重策略 |
| `splitter.py` | 递归切分、块内重叠、最小长度过滤 | 向量化 |
| `embeddings.py` | Ollama 与离线哈希两套实现、L2 归一化、可用性探测 | 检索排序 |
| `vector_store.py` | 向量矩阵、余弦检索、持久化、集合管理、相邻块查询 | 文本解析 |
| `llm.py` | Ollama / OpenAI 兼容客户端、流式解析 | 拼 prompt |
| `rag_chain.py` | 检索编排、上下文预算、prompt、降级、事件流 | 具体 UI 渲染 |
| `ingest.py` | 入库事务：解析 → 切分 → 向量化 → 写库 → 登记，返回逐文件报告 | 检索 |
| `text_utils.py` | 展示层文本清洗、截断预览、查询与原文共有短语的高亮 | 检索与生成 |
| `conversations.py` | 按集合读写对话 JSON（过滤非法消息、上限 200 条、原子写） | 生成回答 |
| `archive.py` | 集合 zip 导出/导入、白名单校验、导入前 `VectorStore.load` 验证 | 增量同步 |
| `cli.py` | 参数解析、输出格式化 | 业务逻辑 |
| `app.py` | 会话状态、上传、进度、来源展示、对话持久化、文档与集合管理 | 业务逻辑（全部复用上面模块） |

依赖方向是单向的：`app.py` / `cli.py` → `rag_chain.py` / `ingest.py` → 其余底层模块。
底层模块之间不允许反向依赖 UI，这也是测试能不启动界面就跑完的原因。

## 数据流

### 入库

```
文件路径
  → resolve_inputs()            过滤隐藏文件、临时文件、不支持的后缀，去重
  → file_sha256()              用于判断"这个文件是不是已经入库过"
  → load_file()                按格式解析成 Document 列表（PDF 一页一个 Document）
  → split_documents()          切块，继承 doc_id / page / 文件元信息
  → embed_documents()          批量向量化（默认每批 16 条）
  → VectorStore.add()          追加到矩阵，维度必须与库一致
  → store.save() + documents.json
```

一次入库是"按文件"提交的：单个文件解析失败只记进 `report.errors`，不影响其它文件，
也不会把半截数据写进向量库（向量与 chunk 是同一次 `add()` 写入的）。

### 问答

```
问题
  → embed_query()              单条向量化
  → store.search()             top_k 余弦召回（语义模式按绝对阈值过滤）
  → 相对阈值过滤              仅离线词法模式：丢掉明显低于最高分的结果
  → neighbors()                命中块前后各取 expand_neighbors 个块补全被切断的上下文
  → format_context()           按 [1][2]... 拼接，受 llm.max_context_chars 约束
  → build_messages()           system 规则 + 最近 history_turns 轮 + 带资料的当前提问
  → llm.chat()                 非流式返回整段，流式逐段 yield
  → Answer(text, sources, hits, elapsed, used_llm)
```

编号一致性由 `format_context()` 唯一负责：它返回的 `sources` 与拼进 prompt 的 `[编号]` 是同一次
enumerate 的产物。上层不允许自己再编号，否则回答里的 `[3]` 可能对应到界面上第 2 条来源。

## 关键数据结构

```python
Document(text, source, doc_id, page, meta)     # 一页或一份文件
Chunk(id, text, source, doc_id, chunk_index, page, meta)   # 检索单元
SearchHit(chunk, score)                        # 一次命中
Answer(question, text, sources, hits, elapsed, used_llm)
```

`doc_id = sha256(文件字节)[:12]`，因此同一文件改名再导入仍是同一个 `doc_id`，去重口径天然一致。
`chunk id = sha1(doc_id:page:index:文本前 64 字)`，用于检索结果去重（相邻块扩展会产生重复候选）。

## 落盘格式

```
data/collections/<集合>/
├── chunks.jsonl     # 每行一个 Chunk.to_dict()，UTF-8，不转义中文
├── vectors.npy      # float32 [n, dim]，已 L2 归一化
├── meta.json        # name / chunks / dim / documents / updated_at
└── documents.json   # 文档登记：doc_id、source、sha256、chunks、pages、ingested_at、embedding

data/conversations/<集合>.json   # role/content/sources/query，最多保留 200 条
```

界面把一次提问拆成两段执行：先把用户消息写入 `messages` 并落盘，再 rerun 一次生成回答。
这样空态引导会在生成开始前退场，流式输出时不会残留示例按钮；代价是多一次很轻的 rerun。

写入使用"临时文件 + 原子替换"，避免进程中断留下半个文件。读取时若 chunk 数与向量行数不匹配，
直接抛 `VectorStoreError` 并提示"删库重建"，而不是带着错位的数据继续检索。

`meta.json` 里的 `dim` 是硬约束：换了 embedding 模型（维度变了）必须新建集合，
否则 `add()` 会在第一行数据就报错，不会静默混入两套向量。

## 检索的数学

向量已归一化，余弦相似度退化为点积：

```
score(q, d) = q · d / (||q|| ||d||)  →  q̂ · d̂   （两者均为单位向量）
```

- **语义模式**：直接用 `retrieval.score_threshold` 做绝对阈值。nomic-embed-text 下相关段落通常在 0.4 以上。
- **离线词法模式**：哈希向量的分数整体低一个量级（实测相关段 0.05~0.13），绝对阈值会误杀全部结果，
  因此改用相对阈值：保留 `score >= 最高分 × lexical_relative_cutoff` 的结果，至少保留 1 条。

离线向量的构造：中文取相邻二字组合，英文取长度 ≥2 的单词，带符号哈希到固定维度，
词频用 `1 + ln(count)` 压制长文本；整个矩阵做 L2 归一化。它的用途是"断网也能跑通、CI 不下载模型"，
不代表语义能力。

## 降级矩阵

| 故障 | 行为 | 用户看到什么 |
| --- | --- | --- |
| Ollama 未启动 / 模型未拉取（embedding） | `build_embeddings` 回退到哈希向量 | 日志一条 warning，功能继续 |
| Ollama 未启动 / 模型未拉取（生成） | `RAGEngine` 捕获 `LLMError`，改走摘录 | 回答末尾一行"模型调用失败：…" |
| 配置 `llm.provider: none` | 不构造客户端，直接摘录 | 回答开头标注"离线提取模式" |
| 向量维度不一致 | `VectorStore.add()` 抛错 | "更换 embedding 模型后请新建集合" |
| 集合为空 | 不检索 | 提示先上传文档 |
| 没有命中 | 不调用模型（省一次推理） | 提示换说法、调低阈值 |

## 扩展点

1. **加 OCR**：在 `loader._load_pdf` 里对 `extract_text()` 为空的页调用 PaddleOCR，返回仍是 `Document`，
   下游无需改动。
2. **换向量库**：实现 `search()` / `neighbors()` / `add()` / `save()` 四个方法即可替换 `VectorStore`。
3. **加 rerank**：在 `RAGEngine.retrieve()` 拿到 `hits` 之后插入一次交叉编码器重排，接口不变。
4. **加混合检索**：`retrieve()` 里把 BM25 结果与向量结果做 RRF 融合，离线模式下收益最明显。
5. **换模型服务**：`llm.build_llm()` 增加一个分支即可，`LLMClient.chat(messages, stream)` 是唯一契约。

## 不变量（改代码时请守住）

1. `format_context()` 是编号的唯一来源。
2. 向量库中 chunk 数量必须等于向量行数。
3. `add()` 前先校验维度，绝不静默截断或填充。
4. 任何模型调用失败都必须有可读降级路径，不允许把 traceback 抛到界面。
5. 入库按文件提交，失败文件不产生半截数据。
