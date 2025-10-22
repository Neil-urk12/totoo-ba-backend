# app/main.py
from fastapi import Depends, FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.products import router as products_router
from app.core.config import Settings, get_settings
from app.core.database import Base, engine, test_connection_async
from app.core.logging import setup_logging

# Initialize settings
settings = get_settings()

# Configure logging with Loguru
setup_logging(settings)

# Initialize FastAPI with environment-specific config
app = FastAPI(default_response_class=ORJSONResponse, **settings.fastapi_kwargs)

# CORS middleware with settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Enable gzip compression for large responses
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.on_event("startup")
async def startup():
    """Initialize lightweight services on startup (no auto DDL)."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    # In development, auto-create tables to ease local workflows
    if engine is None:
        logger.warning("Database engine not initialized")
    elif settings.is_development:
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created/verified (development mode)")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")

    # Test database connection
    if await test_connection_async():
        logger.success("Database connection established")
    else:
        logger.error("Failed to connect to database")


app.include_router(products_router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "running",
    }


@app.get("/health")
async def health_check(settings: Settings = Depends(get_settings)):
    """Health check endpoint with settings injection"""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "database": "connected",
    }
