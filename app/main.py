# app/main.py
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import Settings, get_settings
from app.core.database import engine, Base
import asyncio

# Initialize settings
settings = get_settings()

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
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print(f"🚀 {settings.app_name} v{settings.app_version}")
    print(f"🌍 Environment: {settings.environment}")
    print(f"🔧 Debug Mode: {settings.debug}")
    print(f"📊 Database: {settings.database_url.split('@')[1]}")  # Hide credentials

# Include routers
# Commenting out products.router as the file is currently empty
# app.include_router(products.router, prefix=settings.api_prefix)

@app.get("/")
async def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "running"
    }

@app.get("/health")
async def health_check(settings: Settings = Depends(get_settings)):
    """Health check endpoint with settings injection"""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "database": "connected"
    }
