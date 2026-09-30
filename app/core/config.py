# app/core/config.py
from functools import lru_cache

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    Configuration priority (highest to lowest):
    1. Environment variables
    2. .env file
    3. Default values

    Environment variables use uppercase with prefix:
    e.g., APP_NAME -> app_name
    """

    # ============================================================================
    # ENVIRONMENT & APPLICATION INFO
    # ============================================================================
    app_name: str = "AI RAG Product Checker"
    app_version: str = "1.0.0"
    environment: str = Field(
        default="development",
        description="Current environment: development, staging, production",
    )
    debug: bool = Field(default=True, description="Enable debug mode")

    # ============================================================================
    # API CONFIGURATION
    # ============================================================================
    api_prefix: str = "/api/v1"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"

    # ============================================================================
    # DATABASE CONFIGURATION
    # ============================================================================
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/product_checker",
        description="Async PostgreSQL connection string",
    )

    # ============================================================================
    # CORS CONFIGURATION
    # ============================================================================
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:8000"],
        description="Allowed CORS origins",
    )
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    # ============================================================================
    # LOGGING CONFIGURATION (LOGURU)
    # ============================================================================
    log_level: str = Field(
        default="INFO",
        description="Global logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    log_level_console: str | None = Field(
        default=None,
        description="Console log level override. Falls back to log_level if not set.",
    )
    log_level_file: str | None = Field(
        default=None,
        description="File log level override. Falls back to log_level if not set.",
    )
    log_file: str | None = Field(
        default=None, description="Log file path. None = stdout only"
    )
    log_format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Loguru log format string",
    )
    log_rotation: str = Field(
        default="500 MB",
        description="Log rotation trigger: size (e.g., '500 MB') or time (e.g., '1 week', '00:00')",
    )
    log_retention: str = Field(
        default="10 days",
        description="Log file retention period (e.g., '10 days', '1 month')",
    )
    log_compression: str = Field(
        default="zip",
        description="Compression format for rotated logs: 'zip', 'gz', 'tar.gz', or empty for none",
    )
    log_serialize: bool = Field(
        default=False,
        description="Serialize logs to JSON format (useful for log aggregation systems)",
    )
    log_backtrace: bool = Field(
        default=True,
        description="Enable full exception traceback logging",
    )
    log_diagnose: bool = Field(
        default=False,
        description="Enable variable values in exception traces (disable in production)",
    )

    # ============================================================================
    # VALIDATORS
    # ============================================================================
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Ensure environment is one of the allowed values"""
        allowed = ["development", "staging", "production"]
        if v.lower() not in allowed:
            raise ValueError(f"Environment must be one of: {', '.join(allowed)}")
        return v.lower()

    @field_validator("log_level", "log_level_console", "log_level_file")
    @classmethod
    def validate_log_level(cls, v: str | None) -> str | None:
        """Ensure log level is valid"""
        if v is None or v == "":
            return None
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of: {', '.join(allowed)}")
        return v.upper()

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Ensure database URL uses asyncpg driver"""
        if not v.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "Database URL must use asyncpg driver: "
                "postgresql+asyncpg://user:password@host:port/dbname"
            )
        return v

    # ============================================================================
    # COMPUTED PROPERTIES
    # ============================================================================
    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == "production"

    @computed_field
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == "development"

    @computed_field
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from environment variable or list"""
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(",")]
        return self.cors_origins

    @computed_field
    @property
    def effective_console_log_level(self) -> str:
        """Get effective console log level (with fallback to global log_level)"""
        return self.log_level_console or self.log_level

    @computed_field
    @property
    def effective_file_log_level(self) -> str:
        """Get effective file log level (with fallback to global log_level)"""
        return self.log_level_file or self.log_level

    @computed_field
    @property
    def fastapi_kwargs(self) -> dict:
        """FastAPI initialization arguments based on environment"""
        kwargs = {
            "title": self.app_name,
            "version": self.app_version,
            "debug": self.debug,
            "docs_url": self.docs_url,
            "redoc_url": self.redoc_url,
            "openapi_url": self.openapi_url,
        }

        # Disable docs in production for security
        if self.is_production:
            kwargs.update(
                {
                    "docs_url": None,
                    "redoc_url": None,
                    "openapi_url": None,
                }
            )

        return kwargs

    # ============================================================================
    # MODEL CONFIGURATION
    # ============================================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",  # No prefix, directly use variable names
        case_sensitive=False,  # Environment variables are case-insensitive
        extra="ignore",  # Ignore extra environment variables
        validate_default=True,  # Validate default values
        str_strip_whitespace=True,  # Strip whitespace from string values
    )


# ============================================================================
# DEPENDENCY INJECTION PATTERN
# ============================================================================
@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance for dependency injection.

    Using lru_cache ensures settings are loaded once and reused.
    To reset cache (e.g., in tests): get_settings.cache_clear()

    Usage in FastAPI endpoints:
    ```
    from fastapi import Depends
    from app.core.config import Settings, get_settings

    @app.get("/info")
    async def info(settings: Settings = Depends(get_settings)):
        return {"app_name": settings.app_name}
    ```
    """
    return Settings()


# ============================================================================
# CONVENIENCE INSTANCE
# ============================================================================
settings = get_settings()
