# app/core/config.py
from functools import lru_cache
from typing import List, Optional
from pydantic import Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
import secrets


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
    database_echo: bool = Field(default=False, description="Log all SQL statements")
    database_pool_size: int = Field(
        default=20, ge=1, le=100, description="Database connection pool size"
    )
    database_max_overflow: int = Field(
        default=10, ge=0, le=50, description="Maximum overflow connections"
    )
    database_pool_timeout: int = Field(
        default=30, ge=1, description="Connection pool timeout in seconds"
    )
    database_pool_recycle: int = Field(
        default=3600, ge=300, description="Recycle connections after N seconds"
    )

    # ============================================================================
    # SECURITY SETTINGS
    # ============================================================================
    secret_key: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret key for JWT encoding - MUST be overridden in production",
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(
        default=60 * 24,  # 24 hours
        ge=1,
        description="JWT token expiration time in minutes",
    )

    # ============================================================================
    # CORS CONFIGURATION
    # ============================================================================
    cors_origins: List[str] = Field(
        default=["http://localhost:5173", "http://localhost:8000"],
        description="Allowed CORS origins",
    )
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]

    # ============================================================================
    # FDA SCRAPER CONFIGURATION
    # ============================================================================
    fda_base_url: str = "https://verification.fda.gov.ph"
    fda_timeout: int = Field(
        default=30, ge=5, le=120, description="HTTP request timeout in seconds"
    )
    fda_max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum retry attempts for failed requests"
    )
    fda_rate_limit_delay: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Delay between scraping requests in seconds",
    )
    fda_max_pages_per_run: int = Field(
        default=10, ge=1, le=1000, description="Maximum pages to scrape per run"
    )

    # ============================================================================
    # BUSINESS DATABANK CONFIGURATION
    # ============================================================================
    business_databank_url: str = "https://databank.business.gov.ph"
    sec_api_url: str = "https://portal.sec.gov.ph"
    sec_api_key: Optional[str] = Field(
        default=None, description="SEC API key for business verification"
    )

    # ============================================================================
    # FUZZY MATCHING CONFIGURATION
    # ============================================================================
    fuzzy_match_threshold: int = Field(
        default=80,
        ge=0,
        le=100,
        description="Minimum similarity score for fuzzy matching (0-100)",
    )
    fuzzy_match_limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of fuzzy match results to return",
    )

    # ============================================================================
    # CACHING CONFIGURATION
    # ============================================================================
    cache_enabled: bool = Field(default=True, description="Enable in-memory caching")
    cache_ttl_minutes: int = Field(
        default=30, ge=1, le=1440, description="Cache time-to-live in minutes"
    )
    cache_max_size: int = Field(
        default=1000, ge=10, le=100000, description="Maximum number of cached items"
    )

    # ============================================================================
    # BACKGROUND TASKS CONFIGURATION
    # ============================================================================
    background_task_timeout: int = Field(
        default=300, ge=10, le=3600, description="Background task timeout in seconds"
    )

    # ============================================================================
    # LOGGING CONFIGURATION
    # ============================================================================
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    log_file: Optional[str] = Field(
        default=None, description="Log file path. None = stdout only"
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

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid"""
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
    def database_url_sync(self) -> str:
        """Synchronous database URL for Alembic migrations"""
        return self.database_url.replace("+asyncpg", "")

    @computed_field
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from environment variable or list"""
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(",")]
        return self.cors_origins

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
@lru_cache()
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
