"""Configuration loader and Pydantic models for LectureNotes AI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env variables into environment
load_dotenv()

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


class EmbeddingConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"


class SegmentationConfig(BaseModel):
    window_size: int = Field(3, ge=1)
    step: int = Field(1, ge=1)
    threshold: float = Field(0.25, ge=0.0, le=1.0)


class KeyBERTConfig(BaseModel):
    top_n: int = Field(8, ge=1)
    diversity: float = Field(0.5, ge=0.0, le=1.0)


class SpacyConfig(BaseModel):
    model: str = "en_core_web_sm"


class NLIConfig(BaseModel):
    model: str = "cross-encoder/nli-deberta-v3-base"
    entailment_threshold: float = Field(0.70, ge=0.0, le=1.0)


class LLMConfig(BaseModel):
    provider: Literal["azure", "claude", "ollama", "gemini", "mock", "openai"] = "gemini"
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    azure_deployment: str = "gpt-4o"
    azure_api_version: str = "2024-02-01"
    azure_endpoint: str = ""
    azure_api_key: str = ""
    claude_model: str = "claude-3-5-sonnet-20241022"
    claude_api_key: str = ""
    max_tokens: int = Field(2048, ge=256)
    temperature: float = Field(0.2, ge=0.0, le=1.0)


class GenerationConfig(BaseModel):
    flashcard_count: int = Field(15, ge=5, le=30)
    mcq_count: int = Field(5, ge=1, le=20)


class IngestConfig(BaseModel):
    pdf_min_chars_per_page: int = Field(10, ge=1)


class CacheConfig(BaseModel):
    directory: str = ".cache/diskcache"
    size_limit_gb: int = Field(2, ge=1)


class AppConfig(BaseModel):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    keybert: KeyBERTConfig = Field(default_factory=KeyBERTConfig)
    spacy: SpacyConfig = Field(default_factory=SpacyConfig)
    nli: NLIConfig = Field(default_factory=NLIConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load config.yaml and merge with environment variables."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    config = AppConfig(**data)

    # Sync Streamlit Cloud secrets into os.environ if running inside Streamlit
    try:
        import streamlit as _st
        if hasattr(_st, "secrets"):
            for _k in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"]:
                if _k in _st.secrets and _k not in os.environ:
                    os.environ[_k] = str(_st.secrets[_k])
    except Exception:
        pass

    # Inject environment variables for sensitive API keys
    if "AZURE_OPENAI_API_KEY" in os.environ and not config.llm.azure_api_key:
        config.llm.azure_api_key = os.environ["AZURE_OPENAI_API_KEY"]
    if "AZURE_OPENAI_ENDPOINT" in os.environ and not config.llm.azure_endpoint:
        config.llm.azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    if "ANTHROPIC_API_KEY" in os.environ and not config.llm.claude_api_key:
        config.llm.claude_api_key = os.environ["ANTHROPIC_API_KEY"]
    if "GEMINI_API_KEY" in os.environ and not config.llm.gemini_api_key:
        config.llm.gemini_api_key = os.environ["GEMINI_API_KEY"]
    elif "GOOGLE_API_KEY" in os.environ and not config.llm.gemini_api_key:
        config.llm.gemini_api_key = os.environ["GOOGLE_API_KEY"]
    if "OPENAI_API_KEY" in os.environ and not config.llm.openai_api_key:
        config.llm.openai_api_key = os.environ["OPENAI_API_KEY"]

    return config
