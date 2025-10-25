#!/usr/bin/env python3
"""
Production server runner with performance optimizations.

This script configures uvloop and httptools for maximum performance
in production environments.
"""

import uvicorn

from app.core.config import get_settings


def run_server():
    """Run the FastAPI server with optimal configuration."""
    settings = get_settings()

    # Base configuration
    config = {
        "app": "app.main:app",
        "host": "0.0.0.0",
        "port": 6769,
        "access_log": True,
        "log_level": settings.log_level.lower(),
    }

    # Environment-specific configurations
    if settings.environment == "development":
        # Development mode with hot reload
        config.update({
            "reload": True,
            "reload_dirs": ["app"],
            "workers": 1,
        })
    else:
        # Production optimizations
        config.update({
            "loop": "uvloop",  # Use uvloop event loop for better performance
            "http": "httptools",  # Use httptools HTTP parser
            "workers": 4,  # Adjust based on CPU cores (2 * num_cores + 1)
            "reload": False,
            "access_log": False,  # Disable for better performance
        })

    uvicorn.run(**config)

if __name__ == "__main__":
    run_server()
