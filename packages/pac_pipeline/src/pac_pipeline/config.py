# packages/pac_pipeline/src/pac_pipeline/config.py
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="PAC_")

    data_dir: Path = Field(default=Path("./data"))
    qdrant_path: Path = Field(default=Path("./data/qdrant"))
    tantivy_path: Path = Field(default=Path("./data/tantivy"))
    duckdb_path: Path = Field(default=Path("./data/catalog/pac.duckdb"))
    models_dir: Path = Field(default=Path("./data/models"))
    lit_binary: str = "lit"

    anthropic_api_key: str = Field(alias="ANTHROPIC_API_KEY", default="")

    embedding_model: str = "BAAI/bge-large-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    extraction_model: str = "claude-haiku-4-5-20251001"
    generation_model: str = "claude-sonnet-4-6"


settings = Settings()
