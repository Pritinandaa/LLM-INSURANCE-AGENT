"""
Configuration management using Pydantic Settings.
Loads environment variables from .env file with validation.
"""

from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Vertex AI Configuration
    vertex_project_id: str = Field(
        ...,
        env="VERTEX_PROJECT_ID",
        description="Google Cloud Project ID"
    )
    vertex_location: str = Field(
        default="us-central1",
        env="VERTEX_LOCATION",
        description="Vertex AI Location"
    )
    gemini_model_id: str = Field(
        default="gemini-2.0-flash",
        env="GEMINI_MODEL_ID",
        description="Gemini Model ID"
    )
    vertex_embedding_model: str = Field(
        default="text-embedding-004",
        env="VERTEX_EMBEDDING_MODEL",
        description="Vertex AI Embedding Model ID"
    )

    # MongoDB Configuration
    mongodb_uri: str = Field(
        ...,
        description="MongoDB connection URI"
    )
    mongodb_database: str = Field(
        default="insurance_underwriting",
        description="MongoDB database name"
    )

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # Logging
    log_level: str = Field(default="INFO")

    # SSL Configuration (for corporate environments with proxy/firewall)
    ssl_verify: bool = Field(
        default=True,
        description="Set to False to disable SSL verification (corporate proxies)"
    )

    # Paths (computed)
    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent

    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return self.project_root / "data"

    @property
    def quotes_dir(self) -> Path:
        """Get quotes output directory path."""
        quotes_path = self.project_root / "quotes"
        quotes_path.mkdir(exist_ok=True)
        return quotes_path

    class Config:
        env_file = Path(__file__).parent.parent / ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()
