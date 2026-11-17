"""界面冒烟测试：用 Streamlit 官方 AppTest 真正执行一遍 app.py。

它会在无头环境里跑完整脚本，因此能抓到"页面能打开但脚本报错"这类问题。
"""
from __future__ import annotations

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit", reason="未安装 streamlit 时跳过界面测试")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_runs_without_exception():
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.run()
    assert not app.exception, f"页面脚本抛异常：{app.exception}"
    # 页面切换（问答/文档/设置/关于）应该存在
    assert len(app.radio) >= 1
    assert set(app.radio[0].options) == {"问答", "文档", "设置", "关于"}
