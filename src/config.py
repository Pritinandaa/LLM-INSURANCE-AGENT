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

    # Google Vertex AI Configuration
    google_cloud_project: str = Field(
        default="",
        description="Google Cloud Project ID for Vertex AI"
    )
    google_cloud_region: str = Field(
        default="us-central1",
        description="Google Cloud Region for Vertex AI"
    )
    google_application_credentials: str = Field(
        default="",
        description="Path to Google service account key file"
    )

    # Vertex AI Specific
    vertex_llm_model: str = Field(
        default="gemini-2.0-flash-exp",
        description="LLM model for Vertex AI"
    )
    vertex_project_id: str = Field(
        default="",
        description="Vertex AI Project ID"
    )
    vertex_location: str = Field(
        default="us-central1",
        description="Vertex AI Location"
    )

    # MongoDB Configuration (Optional)
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017",
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
        env_file = ".env"
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
