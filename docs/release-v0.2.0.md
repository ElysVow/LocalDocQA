# LocalDocQA v0.2.0

完全本地运行的文档问答工具：PDF / Word / Markdown 入库，Ollama 本地推理，回答带可展开的来源。

## 这个版本包含

- Streamlit 界面：空态引导、示例问题、来源按查询词高亮、对话按集合落盘。
- 多集合管理：文档详情、逐文件入库报告、集合 zip 导入/导出。
- CLI + Web 双入口；没有 Ollama 时自动降级到离线哈希向量，流程仍可演示。
- 检索评估脚本与 8 题评估集；69 项 pytest、ruff 与 GitHub Actions CI。

## 快速开始

```bash
git clone https://github.com/ElysVow/LocalDocQA.git
cd LocalDocQA
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -e ".[ui,dev]"
copy config.example.yaml config.yaml
python -m streamlit run app.py
```

没有本地模型也可以先运行：

```bash
python scripts/smoke_test.py
```

## 验证

```bash
python -m pytest -q
python -m ruff check .
python scripts/smoke_test.py
```

## 已知限制

扫描版 PDF 暂不做 OCR；纯向量召回对缩写、编号类查询不够稳；7B 模型在多跳问题上可能漏步骤。下一阶段计划见 `docs/ROADMAP.md`。
