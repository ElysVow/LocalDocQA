# Changelog

本项目的重要变更记录在此。版本号遵循语义化版本。

## [0.2.0] - 2026-11-17

首个公开发布版本，定位是“完全在本机运行的文档问答工具”。

### Added

- Streamlit 界面：空态引导、示例问题、来源展开与原文高亮、对话导出。
- 多集合管理：新建、删除、按集合保存对话记录；刷新或重启后可继续上次对话。
- 文档管理：文档详情、逐文件入库报告、集合 zip 导入与导出。
- 设置页：配置查看、模型探测缓存、数据占用统计、批量清理对话。
- CLI 与 Web 双入口；离线哈希向量自动降级，保证没有 Ollama 时也能跑通。
- 检索评估脚本与 8 题评估集。

### Changed

- 精简界面技术噪音，修正宽屏/窄屏布局与对话贴底行为。
- 项目署名、GitHub 链接、clone 地址与 Git 作者统一为 `ElysVow`。
- CI 覆盖 Python 3.10–3.13，增加 ruff 与端到端冒烟任务。

### Validation

- `python -m pytest -q`：69 项通过。
- `python -m ruff check .`：All checks passed。
- `python scripts/smoke_test.py`：通过，包含 `provider=ollama -> hash` 降级路径。

[0.2.0]: https://github.com/ElysVow/LocalDocQA/releases/tag/v0.2.0
