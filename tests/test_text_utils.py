from __future__ import annotations

from localdocqa.text_utils import clean_text, highlight, preview


def test_clean_text_strips_headings_and_collapses_space():
    assert clean_text("## 实验报告\n\n要交   三部分") == "实验报告 要交 三部分"


def test_preview_truncates_with_ellipsis():
    result = preview("a" * 300, max_chars=50)
    assert result.endswith("…")
    assert len(result) == 51


def test_preview_keeps_short_text():
    assert preview("短文本") == "短文本"


def test_highlight_marks_shared_phrase():
    html_text = highlight("每次实验需要提交三部分：实验报告 PDF", "实验报告要交什么？")
    assert "<mark>实验报告</mark>" in html_text


def test_highlight_escapes_html():
    html_text = highlight("<script>x</script> 实验报告", "实验报告")
    assert "<script>" not in html_text
    assert "&lt;script&gt;" in html_text
    assert "<mark>实验报告</mark>" in html_text


def test_highlight_without_shared_phrase():
    assert highlight("完全无关的内容", "实验报告要交什么？") == "完全无关的内容"
