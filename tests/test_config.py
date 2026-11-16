from __future__ import annotations

from pathlib import Path

from localdocqa.config import DEFAULTS, Config


def test_defaults_load_without_file(tmp_path):
    config = Config.load(root=tmp_path)
    assert config.get("llm.provider") == "ollama"
    assert config.get("chunking.size") == DEFAULTS["chunking"]["size"]
    assert config.source is None


def test_deep_merge_keeps_other_defaults(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "llm:\n  model: qwen2.5:3b\nretrieval:\n  top_k: 3\n", encoding="utf-8"
    )
    config = Config.load(root=tmp_path)
    assert config.get("llm.model") == "qwen2.5:3b"
    assert config.get("llm.base_url") == DEFAULTS["llm"]["base_url"]
    assert config.get("retrieval.top_k") == 3
    assert config.get("retrieval.score_threshold") == DEFAULTS["retrieval"]["score_threshold"]


def test_get_set_dotted(tmp_path):
    config = Config({}, root=tmp_path)
    config.set("app.title", "自定义标题")
    assert config.get("app.title") == "自定义标题"
    assert config.get("app.not_exist", "fallback") == "fallback"


def test_data_dir_created(tmp_path):
    config = Config({}, root=tmp_path)
    data_dir = config.data_dir()
    assert data_dir.is_dir()
    assert data_dir == (tmp_path / "data").resolve()
    assert config.collections_dir().is_dir()


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_BASE_URL", "192.168.1.5:11434")
    monkeypatch.setenv("LOCALDOCQA_DATA_DIR", str(tmp_path / "custom"))
    config = Config.load(root=tmp_path)
    assert config.get("llm.base_url") == "http://192.168.1.5:11434"
    assert config.get("embeddings.base_url") == "http://192.168.1.5:11434"
    assert Path(config.get("app.data_dir")).name == "custom"


def test_save_roundtrip(tmp_path):
    config = Config({}, root=tmp_path)
    config.set("llm.model", "qwen2.5:1.5b")
    path = config.save(tmp_path / "out.yaml")
    reloaded = Config.load(path=path, root=tmp_path)
    assert reloaded.get("llm.model") == "qwen2.5:1.5b"
    assert reloaded.source == path.resolve()
