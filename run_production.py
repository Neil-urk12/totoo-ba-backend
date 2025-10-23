#!/usr/bin/env python3
"""
Production server runner with performance optimizations.

This script configures uvloop and httptools for maximum performance
in production environments.
"""
import sys
import warnings
from pathlib import Path

# Add app directory to Python path
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))

try:
    import uvloop
    uvloop.install()
    print("✓ uvloop installed and configured")
except ImportError:
    warnings.warn("uvloop not available, falling back to default event loop")

import uvicorn
from app.core.config import get_settings

def run_server():
    """Run the FastAPI server with optimal configuration."""
    settings = get_settings()
    
    # Production configuration
    config = {
        "app": "app.main:app",
        "host": "0.0.0.0",
        "port": 6769,
        "loop": "uvloop",  # Use uvloop event loop
        "http": "httptools",  # Use httptools HTTP parser
        "workers": 1,  # Single worker for development, increase for production
        "access_log": True,
        "log_level": settings.log_level.lower(),
    }
    
    # Conditional configurations based on environment
    if settings.environment == "development":
        config.update({
            "reload": True,
            "reload_dirs": ["app"],
        })
    else:
        # Production optimizations
        config.update({
            "workers": 4,  # Adjust based on CPU cores
            "reload": False,
            "access_log": False,  # Disable for better performance
        })
    
    print(f"🚀 Starting {settings.app_name} in {settings.environment} mode")
    print(f"📊 Performance optimizations: uvloop + httptools")
    
    uvicorn.run(**config)

if __name__ == "__main__":
    run_server()
