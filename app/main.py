# app/main.py
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import Settings, get_settings
from app.core.database import engine, Base

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

from app.core.database import test_connection_async


@app.on_event("startup")
async def startup():
    """Initialize database on startup"""
    print(f"🚀 {settings.app_name} v{settings.app_version}")
    print(f"🌍 Environment: {settings.environment}")
    print(f"🔧 Debug Mode: {settings.debug}")
    print(f"📊 Database: {settings.database_url.split('@')[1]}")  # Hide credentials

    # Only try to create tables if we have a valid engine
    if engine is not None:
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            print("✅ Database tables initialized")
        except Exception as e:
            print(f"❌ Failed to initialize database tables: {e}")
    else:
        print("❌ No database engine available - skipping table creation")

    # Test database connection
    if await test_connection_async():
        print("✅ Connected to database successfully")
    else:
        print("❌ Failed to connect to database")


# Include routers
from app.api.products import router as products_router

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
