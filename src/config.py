from typing import List, Literal
import os
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"

class DefaultSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[str(ENV_FILE_PATH), ".env"],
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )


class ArxivSettings(DefaultSettings):
    """arXiv API client settings."""

    model_config = SettingsConfigDict(
        env_file=[str(ENV_FILE_PATH), ".env"],
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    base_url: str = "https://export.arxiv.org/api/query"
    pdf_cache_dir: str = "./data/arxiv_pdfs"
    rate_limit_delay: float = 3.0  # seconds between requests
    timeout_seconds: int = 30
    max_results: int = 15
    search_category: str = "cs.AI"  # Default category to search
    download_max_retries: int = 3
    download_retry_delay_base: float = 5.0  # Base delay for exponential backoff on download retries
    max_concurrent_downloads: int = 5  # Limit concurrent downloads to avoid overwhelming the server
    max_concurrent_parsing: int = 1  # Limit concurrent parsing to manage resource usage
    
    namespaces: dict = Field(
        default={
            "atom": "http://www.w3.org/2005/Atom",
            "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
    )
    @field_validator("pdf_cache_dir", mode="before")
    @classmethod
    def validate_cache_dir(cls,v:str)-> str:
        os.makedirs(v, exist_ok=True)
        return v


class PDFParserSettings(DefaultSettings):
    """PDF parser service settings."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="PDF_PARSER__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    max_pages: int = 30
    max_file_size_mb: int = 20
    do_ocr: bool = False
    do_table_structure: bool = True


class OpenSearchSettings(DefaultSettings):
    """OpenSearch client settings."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="OPENSEARCH__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "http://localhost:9200"
    index_name: str = "arxiv-papers"
    max_text_size: int = 10000

class Settings(DefaultSettings):
    """Application settings."""

    app_version: str = "0.1.0"
    debug: bool = True
    environment: Literal["development", "staging", "production"] = "development"
    service_name: str = "rag-api"

    postgres_database_url: str = "postgresql://rag_user:rag_password@localhost:5432/rag_db"
    postgres_echo_sql: bool = False
    postgres_pool_size: int = 20
    postgres_max_overflow: int = 0

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    ollama_timeout: int = 300

    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    pdf_parser: PDFParserSettings = Field(default_factory=PDFParserSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)

    @field_validator("postgres_database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not (v.startswith("postgresql://") or v.startswith("postgresql+psycopg2://")):
            raise ValueError("Database URL must start with 'postgresql://' or 'postgresql+psycopg2://'")
        return v



def get_settings() -> Settings:
    """Get application settings."""
    return Settings()