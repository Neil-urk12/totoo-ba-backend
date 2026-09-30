"""Shared HTML parsing, cleanup, retry, and transaction helpers for food imports."""

import asyncio
import re
import traceback
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from sqlalchemy.exc import OperationalError

from app.core.database import async_session


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry failed operations"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (OperationalError, ConnectionError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2**attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed")
            raise last_exception

        return wrapper

    return decorator


def extract_data_from_html(file_path: str) -> pd.DataFrame:
    """
    Extract all rows from HTML table and return as pandas DataFrame.
    Handles various edge cases and encoding issues.

    Args:
        file_path: Path to the HTML file containing the table

    Returns:
        DataFrame with extracted data
    """
    file = Path(file_path)
    logger.info(f"📂 Processing file: {file.name}")

    if not file.exists():
        logger.error(f"File does not exist: {file_path}")
        raise FileNotFoundError(f"File does not exist: {file_path}")

    file_size = file.stat().st_size
    if file_size == 0:
        logger.warning(f"File is empty: {file_path}")
        return pd.DataFrame()

    logger.info(f"File size: {file_size:,} bytes")

    try:
        content = None
        encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]

        for encoding in encodings:
            try:
                with file.open(encoding=encoding) as f:
                    content = f.read()
                logger.info(f"Successfully read file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            logger.error("Failed to read file with any encoding")
            return pd.DataFrame()

        soup = BeautifulSoup(content, "html.parser")
        tables = soup.find_all("table")

        if not tables:
            logger.warning("No tables found in file")
            return pd.DataFrame()

        logger.info(f"✅ Found {len(tables)} table(s)")

        table = tables[0]
        rows = table.find_all("tr")

        if len(rows) < 2:
            logger.warning("Table has no data rows")
            return pd.DataFrame()

        header_row = rows[0]
        headers = [
            cell.get_text(strip=True) for cell in header_row.find_all(["th", "td"])
        ]
        headers = [h if h else f"Column_{i}" for i, h in enumerate(headers)]

        logger.info(f"📋 Headers: {headers}")
        logger.info(f"📊 Number of columns: {len(headers)}")

        data = []
        skipped_rows = 0

        for row_idx, row in enumerate(rows[1:], start=2):
            try:
                cells = row.find_all(["td", "th"])
                row_data = [cell.get_text(strip=True) for cell in cells]

                if len(row_data) < len(headers):
                    row_data.extend([""] * (len(headers) - len(row_data)))
                elif len(row_data) > len(headers):
                    row_data = row_data[: len(headers)]

                if row_data and any(cell for cell in row_data):
                    data.append(row_data)
                else:
                    skipped_rows += 1
            except Exception as e:
                logger.warning(f"Error processing row {row_idx}: {e}")
                skipped_rows += 1
                continue

        if skipped_rows > 0:
            logger.info(f"Skipped {skipped_rows} empty or invalid rows")

        df = pd.DataFrame(data, columns=headers)

        logger.info(f"✅ Extracted {len(df)} rows")
        logger.info(f"\n📊 First few rows:\n{df.head()}")

        return df

    except Exception as e:
        logger.error(f"❌ Failed to process {file_path}: {e}")
        logger.debug(traceback.format_exc())
        raise


def parse_date_safely(date_str: Any) -> date | None:
    """
    Safely parse various date formats.

    Args:
        date_str: Date string in various formats

    Returns:
        date object or None if parsing fails
    """
    if pd.isna(date_str) or date_str is None or date_str == "":
        return None

    if isinstance(date_str, date):
        return date_str

    date_str = str(date_str).strip()

    date_formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%m-%d-%Y",
        "%d-%m-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%Y%m%d",
    ]

    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str, fmt)  # noqa: DTZ007
            return dt.date()
        except ValueError:
            continue

    try:
        dt = pd.to_datetime(date_str, errors="coerce")
        if pd.notna(dt):
            return dt.date()
    except Exception:
        pass

    logger.warning(f"Could not parse date: {date_str}")
    return None


def clean_string_field(value: Any, max_length: int | None = None) -> str | None:
    """
    Clean and validate string fields with regex support.
    Handles Excel formula notation and other edge cases.

    Args:
        value: Input value
        max_length: Maximum allowed length

    Returns:
        Cleaned string or None
    """
    if pd.isna(value) or value is None:
        return None

    cleaned = str(value).strip()
    cleaned = re.sub(r'^=["\'](.*)["\']$', r"\1", cleaned)
    cleaned = re.sub(r"^=+", "", cleaned)
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.replace("\x00", "").replace("\r", " ").replace("\n", " ")
    cleaned = cleaned.strip('"').strip("'")

    if max_length and len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
        logger.warning(f"Truncated value to {max_length} characters: {cleaned[:50]}...")

    return cleaned if cleaned else None


@asynccontextmanager
async def get_session_with_rollback():
    """Context manager for database session with automatic rollback on error"""
    session = async_session()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Database error, rolling back: {e}")
        raise
    finally:
        await session.close()
