"""
Shared Pydantic-Settings configuration for PipelineMind.
Named pm_config to avoid collision with the third-party 'config' package.

Model update history:
  2024-05-14: llama3-8b-8192 decommissioned by Groq
              → replaced with llama-3.1-8b-instant
              llama3-70b-8192 decommissioned by Groq
              → replaced with llama-3.3-70b-versatile
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Groq
    groq_api_key: str = Field(..., description="Groq Cloud API key")

    # Model tiers — update these when Groq deprecates models
    # Fast/cheap: intent classification, HyDE, chunk summaries
    groq_model_fast: str = "llama-3.1-8b-instant"
    # Strong: fallback LLM classifier (rarely used — keyword fast-path handles most)
    groq_model_strong: str = "llama-3.3-70b-versatile"
    # Agent: function-calling loop — must support tool_use
    groq_model_agent: str = "llama-3.3-70b-versatile"

    # Storage paths
    chroma_path: Path = Path("./data/chroma_db")
    duckdb_path: Path = Path("./data/pipelinemind.db")
    pipeline_repo_path: Path = Path("./data/pipeline_repo")
    bm25_index_path: Path = Path("./data/bm25_index.pkl")
    embed_cache_dir: Path = Path("./data/model_cache")

    # Server
    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000
    streamlit_port: int = 8501

    # Logging
    log_level: str = "INFO"
    environment: str = "development"

    # RAG knobs
    max_context_tokens: int = 6000
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_fused: int = 10
    top_k_rerank: int = 5
    rrf_k: int = 60
    confidence_threshold: float = 0.6
    hyde_enabled: bool = True
    rerank_enabled: bool = True

    # Agent
    agent_max_iterations: int = 5


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()
