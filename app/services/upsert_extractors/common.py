"""Shared parsing, cleanup, loading, and orchestration for food imports."""

import asyncio
import re
import traceback
from collections.abc import Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError, OperationalError

from app.core.database import Base, async_session, engine


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


@retry_on_failure(max_retries=3, delay=1.0)
async def create_tables():
    """Create database tables if they don't exist"""
    logger.info("🔨 Creating database tables...")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        logger.debug(traceback.format_exc())
        raise


@retry_on_failure(max_retries=3, delay=2.0)
async def bulk_upsert(
    model: type,
    key: str,
    fields: Sequence[str],
    data: list[dict[str, Any]],
    batch_size: int = 500,
):
    """
    Insert data into database with conflict resolution (upsert).
    Uses PostgreSQL's ON CONFLICT clause for efficient upserts.
    Includes comprehensive error handling and progress tracking.

    Args:
        data: List of dictionaries containing row data
        batch_size: Number of rows to insert per batch
    """
    logger.info(f"💾 Inserting {len(data)} rows into database...")

    if not data:
        logger.warning("No data to insert")
        return

    total_inserted = 0
    total_failed = 0
    failed_records = []

    try:
        async with get_session_with_rollback() as session:
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(data) + batch_size - 1) // batch_size

                try:
                    # Validate batch data
                    valid_batch = []
                    for idx, record in enumerate(batch):
                        try:
                            # Check for required fields
                            if not record.get(key):
                                logger.warning(
                                    f"Skipping record {i + idx}: missing {key}"
                                )
                                failed_records.append(record)
                                total_failed += 1
                                continue

                            cleaned_record = {
                                field: record.get(field) for field in fields
                            }

                            valid_batch.append(cleaned_record)

                        except Exception as e:
                            logger.warning(f"Error validating record {i + idx}: {e}")
                            failed_records.append(record)
                            total_failed += 1
                            continue

                    if not valid_batch:
                        logger.warning(
                            f"Batch {batch_num} has no valid records, skipping"
                        )
                        continue

                    # PostgreSQL INSERT ... ON CONFLICT (upsert)
                    stmt = insert(model).values(valid_batch)

                    # Update non-key fields on conflict
                    update_stmt = stmt.on_conflict_do_update(
                        index_elements=[key],
                        set_={
                            field: getattr(stmt.excluded, field)
                            for field in fields
                            if field != key
                        },
                    )

                    await session.execute(update_stmt)
                    await session.flush()

                    total_inserted += len(valid_batch)
                    logger.info(
                        f"✅ Batch {batch_num}/{total_batches} processed ({len(valid_batch)} rows)"
                    )

                except IntegrityError as e:
                    logger.error(f"Integrity error in batch {batch_num}: {e}")
                    total_failed += len(batch)
                    failed_records.extend(batch)
                    await session.rollback()
                    continue

                except Exception as e:
                    logger.error(f"Error processing batch {batch_num}: {e}")
                    logger.debug(traceback.format_exc())
                    total_failed += len(batch)
                    failed_records.extend(batch)
                    await session.rollback()
                    continue

        # Log summary
        logger.info(f"\n{'=' * 60}")
        logger.info("🎉 Processing complete!")
        logger.info(f"  ✅ Successfully processed: {total_inserted} rows")
        if total_failed > 0:
            logger.warning(f"  ⚠️  Failed: {total_failed} rows")

            # Save failed records to file for review
            if failed_records:
                failed_file = (
                    f"failed_records_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.csv"
                )
                try:
                    df_failed = pd.DataFrame(failed_records)
                    df_failed.to_csv(failed_file, index=False)
                    logger.info(f"  💾 Failed records saved to: {failed_file}")
                except Exception as e:
                    logger.error(f"Could not save failed records: {e}")
        logger.info(f"{'=' * 60}\n")

    except Exception as e:
        logger.error(f"❌ Critical error during bulk upsert: {e}")
        logger.debug(traceback.format_exc())
        raise


@retry_on_failure(max_retries=3, delay=1.0)
async def verify_insertion(model: type, key: str, label_field: str, limit: int = 5):
    """Verify data was inserted correctly by querying a few rows"""
    logger.info(f"🔍 Verifying insertion (showing {limit} rows)...")

    try:
        async with async_session() as session:
            result = await session.execute(select(model).limit(limit))
            rows = result.scalars().all()

            if rows:
                logger.info(f"✅ Found {len(rows)} rows in database")
                for row in rows:
                    logger.info(f"  - {getattr(row, key)}: {getattr(row, label_field)}")
            else:
                logger.warning("⚠️  No rows found in database")
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        logger.debug(traceback.format_exc())
        raise


@retry_on_failure(max_retries=3, delay=1.0)
async def get_record_count(model: type) -> int:
    """Get total number of records in database"""
    try:
        async with async_session() as session:
            result = await session.execute(select(text("COUNT(*)")).select_from(model))
            count = result.scalar()
            return count or 0
    except Exception as e:
        logger.error(f"Error getting record count: {e}")
        return 0


async def process_files(
    paths: list[Path],
    extract: Callable[[str], pd.DataFrame],
    transform: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    upsert: Callable[..., Awaitable[None]],
    verify: Callable[..., Awaitable[None]],
    count: Callable[[], Awaitable[int]],
    batch: bool = False,
):
    """Extract and load files, continuing after individual file failures."""
    await create_tables()
    if not paths:
        logger.warning("⚠️  No matching files found")
        return

    all_data = []
    successful_files = 0
    failed_files = []
    for idx, file_path in enumerate(paths, 1):
        logger.info(f"Processing file {idx}/{len(paths)}: {file_path.name}")
        try:
            df = extract(str(file_path))
            if df.empty:
                logger.warning(f"⚠️  No data extracted: {file_path.name}")
                failed_files.append((file_path.name, "No data extracted"))
                continue
            df_clean = transform(df)
            if df_clean.empty:
                logger.warning(f"⚠️  No valid data after cleaning: {file_path.name}")
                failed_files.append((file_path.name, "No valid data after cleaning"))
                continue
            all_data.extend(df_clean.to_dict("records"))
            successful_files += 1
            logger.info(f"✅ Successfully processed {file_path.name}")
        except Exception as e:
            logger.error(f"❌ Failed to process {file_path.name}: {e}")
            logger.debug(traceback.format_exc())
            failed_files.append((file_path.name, str(e)))
            continue

    logger.info("📊 File Processing Summary:")
    logger.info(f"  Total files: {len(paths)}")
    logger.info(f"  Successful: {successful_files}")
    logger.info(f"  Failed: {len(failed_files)}")
    for filename, reason in failed_files:
        logger.warning(f"    - {filename}: {reason}")
    if not all_data:
        logger.warning("⚠️  No data extracted from any files")
        return

    logger.info(f"📊 Total rows extracted from all files: {len(all_data)}")
    try:
        await upsert(all_data, batch_size=500)
    except Exception as e:
        logger.error(f"Failed during bulk insert: {e}")
        logger.debug(traceback.format_exc())

    try:
        await verify(limit=10 if batch else 5)
    except Exception as e:
        logger.warning(f"Verification failed: {e}")
        if batch:
            return
    try:
        total = await count()
        logger.info(f"📊 Total records in database: {total}")
    except Exception as e:
        logger.warning(f"Could not get record count: {e}")
