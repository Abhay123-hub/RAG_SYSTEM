from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["openai", "groq"] = "openai"
    openai_api_key: SecretStr = SecretStr("")
    groq_api_key: SecretStr = SecretStr("")
    llm_model: str = "gpt-4o"
    router_model: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    retrieval_k: int = Field(default=15, gt=0)
    rerank_top_n: int = Field(default=5, gt=0)
    score_threshold: float = Field(default=0.5, ge=0, le=1)
    router_confidence_threshold: float = Field(default=0.4, ge=0, le=1)
    max_input_length: int = Field(default=10_000, gt=0)
    history_turns: int = Field(default=5, ge=1)
    log_level: str = "INFO"
    upload_dir: Path = Path("data/uploads")
    chroma_dir: Path = Path("data/chroma_db")
    log_dir: Path = Path("logs")

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()