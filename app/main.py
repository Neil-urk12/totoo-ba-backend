# app/main.py
from fastapi import Depends, FastAPI
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
app = FastAPI(**settings.fastapi_kwargs)

# CORS middleware with settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


@app.on_event("startup")
async def startup():
    """Initialize database on startup"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    # Create tables only in development to avoid startup overhead in prod
    if settings.is_development and engine is not None:
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created/verified (development mode)")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
    elif engine is None:
        logger.warning("Database engine not initialized")

    # Lightweight DB health check (SELECT 1)
    if await test_connection_async():
        logger.success("Database connection healthy")
    else:
        logger.error("Database health check failed")


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
