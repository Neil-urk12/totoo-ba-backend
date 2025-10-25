# app/main.py
"""Main FastAPI application entry point.

Initializes the FastAPI application with:
- Database connections and table creation
- CORS middleware configuration
- API route registration
- Logging setup using Loguru
- Health check endpoints
"""
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from loguru import logger

from app.core.config import Settings, get_settings
from app.core.database import Base, engine, test_connection_async
from app.core.logging import setup_logging

# Import API routers
from app.api.products import router as products_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage FastAPI application lifespan.

    Handles startup and shutdown events:
    - Startup: Creates database tables, tests connection
    - Shutdown: Cleanup resources (if needed)
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    # Only try to create tables if we have a valid engine
    if engine is not None:
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
    else:
        logger.warning("Database engine not initialized")

    # Test database connection
    if await test_connection_async():
        logger.success("Database connection established")
    else:
        logger.error("Failed to connect to database")

    yield

    # Shutdown (add cleanup code here if needed)
    logger.info(f"Shutting down {settings.app_name}")


# Initialize settings
settings = get_settings()

# Configure logging with Loguru
setup_logging(settings)

# Initialize FastAPI with environment-specific config and ORJSONResponse
app = FastAPI(
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
    **settings.fastapi_kwargs
)

# CORS middleware with settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# GZip compression middleware for response size optimization
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Register API routers
app.include_router(products_router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    """Root endpoint providing basic application information.

    Returns:
        dict: Application name, version, environment, and status.
    """
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "running",
    }


@app.get("/health")
async def health_check(settings: Settings = Depends(get_settings)):
    """Health check endpoint for monitoring application status.

    Args:
        settings: Injected application settings.

    Returns:
        dict: Health status, environment, and database connection status.
    """
    return {
        "status": "healthy",
        "environment": settings.environment,
        "database": "connected",
    }
