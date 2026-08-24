"""
Centralized Configuration Module
=================================
Manages all environment variables and platform settings in a single type-safe module.
Industry standards: 12-Factor App design pattern.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # Server Configuration
    API_HOST: str = Field(default="0.0.0.0", description="API server host IP")
    API_PORT: int = Field(default=8000, description="API server port")
    ENVIRONMENT: str = Field(default="development", description="Environment mode: development, testing, production")
    TESTING: bool = Field(default=False, description="Flag for automated test suite execution")
    LOG_LEVEL: str = Field(default="INFO", description="Global logging verbosity level")

    # Storage & Database
    DATABASE_URL: Optional[str] = Field(default=None, description="PostgreSQL connection string")
    REDIS_HOST: str = Field(default="localhost", description="Redis server host")
    REDIS_PORT: int = Field(default=6379, description="Redis server port")
    REDIS_URL: Optional[str] = Field(default=None, description="Redis connection URI")
    ELASTICSEARCH_HOSTS: str = Field(default="http://localhost:9200", description="Elasticsearch cluster URL")

    # MLOps & Experimentation
    MLFLOW_TRACKING_URI: Optional[str] = Field(default=None, description="MLflow tracking server or directory URI")
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Google Gemini API key")

    # Model Parameters
    DEFAULT_TOP_K: int = Field(default=10, ge=1, le=100)
    MAX_CANDIDATE_POOL_SIZE: int = Field(default=100, ge=10, le=1000)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True
    )

    @property
    def is_testing(self) -> bool:
        return self.TESTING or self.ENVIRONMENT.lower() == "testing" or os.getenv("TESTING", "").lower() in ("1", "true", "yes")

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"


# Singleton instance loaded once at startup
settings = Settings()
