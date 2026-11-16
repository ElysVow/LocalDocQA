# 贡献指南

## 环境

```bash
git clone https://github.com/YOUR-NAME/LocalDocQA.git
cd LocalDocQA
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

`[dev]` 会带上 pytest、ruff 和 streamlit（界面测试需要）。

## 提交前必跑的三件事

```bash
ruff check .                  # 静态检查，CI 会卡这一关
pytest                        # 单元测试 + 界面冒烟（AppTest）
python scripts/smoke_test.py   # 端到端：入库 → 检索 → 回答 → 去重 → 降级
```

这三个命令都不需要 GPU、不需要装 Ollama、不下载模型；如果你的改动让它们必须依赖模型才能跑，
说明依赖注入做错了——`build_embeddings` / `build_llm` 都支持外部传入实例，测试里传离线实现即可。

## 代码风格

- 行宽 110，ruff 配置在 `pyproject.toml` 里，`ruff check . --fix` 基本能自动处理导入顺序。
- 注释写"为什么"，不写"做了什么"。例如不要写"循环遍历 chunks"，要写"相邻块可能被切断，这里补全上下文"。
- 新增配置项必须同时更新三处：`config.py` 的 `DEFAULTS`、`config.example.yaml` 的注释、README 的配置表。
- 面向用户的报错必须是可执行的一句话（做什么、在哪做），不要把异常类型直接抛给用户。

## 测试要求

- 修 bug 先写一个能复现的测试，再改代码。
- 新格式解析器至少覆盖：正常文件、空文件、编码异常（GBK）三类用例。
- 涉及检索改动的，跑一次 `python scripts/eval_rag.py -c demo` 并在 PR 里贴 hit@1 / hit@k 前后对比。

## PR 检查清单

- [ ] 三个命令全绿
- [ ] 新增行为有测试覆盖
- [ ] 改动了配置项 / 存储格式 / prompt 的话，同步更新 README 与 docs
- [ ] 没有把 `data/`、模型权重、真实文档提交进仓库（`.gitignore` 已覆盖，提交前 `git status` 看一眼）
- [ ] 涉及隐私边界的改动（新增网络请求、新增遥测）必须在 PR 描述里显式说明

## 特别欢迎的两类贡献

1. **扩充评估集**：给 `examples/eval_set.jsonl` 加问题，附上 `expected_source` 和关键 `keywords`。
   真实使用场景的问题比造出来的更能暴露检索缺陷。
2. **真实格式的解析用例**：脱敏后的 DOCX 表格、双栏 PDF、带页眉页脚的扫描件，放进 `tests/` 作为 fixture。

## 行为准则

讨论对事不对人。拒绝"为了支持某格式把核心链路改得面目全非"的 PR——这类需求走 loader 扩展点。
