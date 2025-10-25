# app/core/logging.py
import logging
import sys
from pathlib import Path

from loguru import logger

from app.core.config import Settings


class InterceptHandler(logging.Handler):
    """
    Intercept standard logging messages and redirect them to Loguru.

    This handler captures logs from third-party libraries (uvicorn, sqlalchemy, etc.)
    that use Python's built-in logging module and forwards them to Loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """
        Process a logging record and forward it to Loguru.

        Args:
            record: The logging record to process
        """
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(settings: Settings) -> None:
    """
    Configure Loguru logging based on application settings.

    This function:
    1. Removes default Loguru handler
    2. Configures console sink with optional colorization and formatting
    3. Configures file sink with rotation, retention, and compression
    4. Intercepts logs from standard logging module (uvicorn, sqlalchemy, etc.)

    Args:
        settings: Application settings containing logging configuration
    """
    # Remove default handler
    logger.remove()

    # =========================================================================
    # CONSOLE SINK
    # =========================================================================
    console_format = (
        settings.log_format
        if not settings.log_serialize
        else "{message}"  # Simple format for JSON
    )

    logger.add(
        sys.stderr,
        format=console_format,
        level=settings.effective_console_log_level,
        colorize=not settings.log_serialize,  # No colors in JSON mode
        serialize=settings.log_serialize,
        backtrace=settings.log_backtrace,
        diagnose=settings.log_diagnose,
    )

    # =========================================================================
    # FILE SINK (if log_file is configured)
    # =========================================================================
    if settings.log_file:
        log_path = Path(settings.log_file)

        # Create log directory if it doesn't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine compression (empty string means no compression)
        compression = settings.log_compression if settings.log_compression else None

        logger.add(
            str(log_path),
            format=settings.log_format,
            level=settings.effective_file_log_level,
            rotation=settings.log_rotation,
            retention=settings.log_retention,
            compression=compression,
            serialize=settings.log_serialize,
            backtrace=settings.log_backtrace,
            diagnose=settings.log_diagnose,
        )

    # =========================================================================
    # INTERCEPT STANDARD LOGGING
    # =========================================================================
    # List of loggers from third-party libraries to intercept
    intercept_loggers = [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
        "sqlalchemy",
        "sqlalchemy.engine",
        "sqlalchemy.pool",
    ]

    # Configure interception
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    for logger_name in intercept_loggers:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False

    # Log the logging configuration
    logger.info(
        f"Logging configured: console={settings.effective_console_log_level}, "
        f"file={settings.effective_file_log_level if settings.log_file else 'disabled'}, "
        f"serialize={settings.log_serialize}"
    )

    if settings.log_file:
        logger.info(
            f"File logging: path={settings.log_file}, "
            f"rotation={settings.log_rotation}, "
            f"retention={settings.log_retention}, "
            f"compression={settings.log_compression or 'none'}"
        )
