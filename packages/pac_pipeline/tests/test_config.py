from __future__ import annotations

from pathlib import Path

import pytest
from pac_pipeline.config import Settings, settings

_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "PAC_DATA_DIR",
    "PAC_QDRANT_PATH",
    "PAC_TANTIVY_PATH",
    "PAC_DUCKDB_PATH",
    "PAC_MODELS_DIR",
    "PAC_LIT_BINARY",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _isolated_settings() -> Settings:
    # Ignore the developer's .env so tests see only the process environment.
    return Settings(_env_file=None)  # type: ignore[call-arg]


class TestDefaults:
    def test_paths(self, clean_env: pytest.MonkeyPatch) -> None:
        s = _isolated_settings()
        assert s.data_dir == Path("./data")
        assert s.qdrant_path == Path("./data/qdrant")
        assert s.tantivy_path == Path("./data/tantivy")
        assert s.duckdb_path == Path("./data/catalog/pac.duckdb")
        assert s.models_dir == Path("./data/models")

    def test_lit_binary(self, clean_env: pytest.MonkeyPatch) -> None:
        assert _isolated_settings().lit_binary == "lit"

    def test_api_key_empty(self, clean_env: pytest.MonkeyPatch) -> None:
        assert _isolated_settings().anthropic_api_key == ""

    def test_model_names(self, clean_env: pytest.MonkeyPatch) -> None:
        s = _isolated_settings()
        assert s.embedding_model == "BAAI/bge-large-en-v1.5"
        assert s.reranker_model == "BAAI/bge-reranker-v2-m3"
        assert s.extraction_model.startswith("claude-haiku")
        assert s.generation_model.startswith("claude-sonnet")


class TestEnvOverrides:
    def test_pac_prefixed_paths(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv("PAC_DUCKDB_PATH", "/tmp/x.duckdb")
        clean_env.setenv("PAC_DATA_DIR", "/tmp/data")
        s = _isolated_settings()
        assert s.duckdb_path == Path("/tmp/x.duckdb")
        assert s.data_dir == Path("/tmp/data")

    def test_lit_binary(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv("PAC_LIT_BINARY", "/opt/bin/lit")
        assert _isolated_settings().lit_binary == "/opt/bin/lit"

    def test_unprefixed_anthropic_key(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert _isolated_settings().anthropic_api_key == "sk-test"


def test_module_singleton_is_settings() -> None:
    assert isinstance(settings, Settings)
