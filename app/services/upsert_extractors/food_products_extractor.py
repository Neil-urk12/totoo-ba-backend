# extractor_to_db_food_products.py
import asyncio
import logging
import os
import re
import traceback
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError, OperationalError

# Import your database configuration
from app.core.database import Base, async_session, engine

# Import the FoodProducts model
from app.models.food_products import FoodProducts

load_dotenv()


# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================
def setup_logging(log_file: str = "food_products_extractor.log"):
    """Configure logging with both file and console output"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ==============================================================================
# RETRY DECORATOR
# ==============================================================================
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


def extract_data_from_file(file_path: str) -> pd.DataFrame:
    """
    Extract data from HTML or CSV files.
    """
    file_ext = Path(file_path).suffix.lower()

    if file_ext == ".csv":
        # Handle CSV files
        logger.info(f"📂 Processing CSV file: {Path(file_path).name}")
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
            logger.info(f"✅ Extracted {len(df)} rows from CSV")
            return df
        except UnicodeDecodeError:
            # Try alternative encodings
            df = pd.read_csv(file_path, encoding="latin-1")
            logger.info(f"✅ Extracted {len(df)} rows from CSV")
            return df
    else:
        # Handle HTML files (existing code)
        return extract_data_from_html(file_path)


# ==============================================================================
# EXTRACT DATA FROM HTML TABLE
# ==============================================================================
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


# ==============================================================================
# DATA TRANSFORMATION & CLEANING
# ==============================================================================
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


def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform the extracted data to match FoodProducts database schema.
    Handles various edge cases and data quality issues.

    Args:
        df: Raw DataFrame from HTML extraction

    Returns:
        Cleaned DataFrame ready for database insertion
    """
    logger.info("🔄 Transforming data...")

    if df.empty:
        logger.warning("Empty DataFrame provided")
        return df

    df_clean = df.copy()

    # Define possible column names for each field (case-insensitive)
    possible_column_names = {
        "registration_number": [
            "Registration Number",
            "Registration No",
            "Registration No.",
            "Reg Number",
            "registration number",
            "registration no",
            "reg number",
            "registrationno",
            "REGISTRATION NUMBER",
            "REGISTRATION NO",
            "REG NUMBER",
            "Reg No",
            "reg no",
            "Certificate Number",
            "certificate number",
            "Cert Number",
            "cert number",
        ],
        "company_name": [
            "Company Name",
            "Company",
            "Applicant",
            "Manufacturer",
            "Applicant Company",
            "company name",
            "company",
            "applicant",
            "manufacturer",
            "applicant company",
            "COMPANY NAME",
            "COMPANY",
            "APPLICANT",
            "MANUFACTURER",
            "APPLICANT COMPANY",
            "Name of Company",
            "name of company",
            "Business Name",
            "business name",
        ],
        "product_name": [
            "Product Name",
            "Product",
            "Name of Product",
            "Generic Name",
            "product name",
            "product",
            "name of product",
            "generic name",
            "PRODUCT NAME",
            "PRODUCT",
            "NAME OF PRODUCT",
            "GENERIC NAME",
            "Item Name",
            "item name",
            "Name",
            "name",
        ],
        "brand_name": [
            "Brand Name",
            "Brand",
            "Trade Name",
            "Trademark",
            "brand name",
            "brand",
            "trade name",
            "trademark",
            "BRAND NAME",
            "BRAND",
            "TRADE NAME",
            "TRADEMARK",
            "Commercial Name",
            "commercial name",
        ],
        "type_of_product": [
            "Type of Product",
            "Product Type",
            "Type",
            "Category",
            "Classification",
            "type of product",
            "product type",
            "type",
            "category",
            "classification",
            "TYPE OF PRODUCT",
            "PRODUCT TYPE",
            "TYPE",
            "CATEGORY",
            "CLASSIFICATION",
            "Product Category",
            "product category",
        ],
        "issuance_date": [
            "Issuance Date",
            "Issue Date",
            "Date of Issue",
            "Date Issued",
            "Issued Date",
            "issuance date",
            "issue date",
            "date of issue",
            "date issued",
            "issued date",
            "issuancedate",
            "issuedate",
            "ISSUANCE DATE",
            "ISSUE DATE",
            "DATE OF ISSUE",
            "DATE ISSUED",
            "ISSUED DATE",
            "Date of Issuance",
            "date of issuance",
        ],
        "expiry_date": [
            "Expiry Date",
            "Expiration Date",
            "Exp Date",
            "Valid Until",
            "Expires",
            "expiry date",
            "expiration date",
            "exp date",
            "valid until",
            "expires",
            "expirydate",
            "expirationdate",
            "EXPIRY DATE",
            "EXPIRATION DATE",
            "EXP DATE",
            "VALID UNTIL",
            "EXPIRES",
            "Validity Date",
            "validity date",
        ],
    }

    # Create a mapping from actual column names to standardized names
    column_mapping = {}

    for standard_name, possible_names in possible_column_names.items():
        for col in df_clean.columns:
            if col in possible_names:
                column_mapping[col] = standard_name
                break

    logger.info(f"📋 Found column mappings: {column_mapping}")

    # Check for missing critical columns
    required_fields = [
        "registration_number",
        "company_name",
        "product_name",
        "brand_name",
        "type_of_product",
        "issuance_date",
        "expiry_date",
    ]

    mapped_fields = set(column_mapping.values())
    missing_fields = set(required_fields) - mapped_fields

    if missing_fields:
        logger.warning(f"⚠️  Missing fields in data: {missing_fields}")
        logger.warning("Available columns: " + ", ".join(df_clean.columns.tolist()))

    # Rename columns based on mapping
    df_clean = df_clean.rename(columns=column_mapping)

    # Add missing columns with None values
    for field in required_fields:
        if field not in df_clean.columns:
            df_clean[field] = None
            logger.warning(f"Added missing column '{field}' with None values")

    # Clean and validate each field
    try:
        # Registration Number - required, must be unique
        if "registration_number" in df_clean.columns:
            df_clean["registration_number"] = df_clean["registration_number"].apply(
                lambda x: clean_string_field(x, max_length=100)
            )
            sample = df_clean["registration_number"].head(10).tolist()
            logger.info(f"Sample cleaned registration numbers: {sample}")

        # Company Name
        if "company_name" in df_clean.columns:
            df_clean["company_name"] = df_clean["company_name"].apply(
                lambda x: clean_string_field(x, max_length=500)
            )

        # Product Name
        if "product_name" in df_clean.columns:
            df_clean["product_name"] = df_clean["product_name"].apply(
                lambda x: clean_string_field(x, max_length=500)
            )

        # Brand Name
        if "brand_name" in df_clean.columns:
            df_clean["brand_name"] = df_clean["brand_name"].apply(
                lambda x: clean_string_field(x, max_length=300)
            )

        # Type of Product
        if "type_of_product" in df_clean.columns:
            df_clean["type_of_product"] = df_clean["type_of_product"].apply(
                lambda x: clean_string_field(x, max_length=200)
            )

        # Date columns
        for date_col in ["issuance_date", "expiry_date"]:
            if date_col in df_clean.columns:
                df_clean[date_col] = df_clean[date_col].apply(parse_date_safely)

    except Exception as e:
        logger.error(f"Error during field cleaning: {e}")
        logger.debug(traceback.format_exc())
        raise

    # Data quality checks and cleaning
    initial_count = len(df_clean)

    # Remove rows with missing registration number (primary key)
    if "registration_number" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["registration_number"])
        logger.info(
            f"Removed {initial_count - len(df_clean)} rows with missing registration_number"
        )

    # Remove duplicate registration numbers, keep first occurrence
    duplicates = df_clean.duplicated(subset=["registration_number"], keep="first")
    if duplicates.any():
        num_duplicates = duplicates.sum()
        logger.warning(
            f"Found {num_duplicates} duplicate registration numbers, keeping first occurrence"
        )
        df_clean = df_clean[~duplicates]

    # Validate dates (expiry should be after issuance)
    if "issuance_date" in df_clean.columns and "expiry_date" in df_clean.columns:
        invalid_dates = (
            df_clean["issuance_date"].notna()
            & df_clean["expiry_date"].notna()
            & (df_clean["expiry_date"] <= df_clean["issuance_date"])
        )
        if invalid_dates.any():
            num_invalid = invalid_dates.sum()
            logger.warning(
                f"Found {num_invalid} rows with expiry_date <= issuance_date"
            )
            invalid_samples = df_clean[invalid_dates][
                ["registration_number", "issuance_date", "expiry_date"]
            ].head()
            logger.warning(f"Examples:\n{invalid_samples}")

    # Check for required fields with None values
    required_not_null = [
        "company_name",
        "product_name",
        "brand_name",
        "type_of_product",
        "issuance_date",
        "expiry_date",
    ]

    for field in required_not_null:
        if field in df_clean.columns:
            null_count = df_clean[field].isna().sum()
            if null_count > 0:
                logger.warning(
                    f"Field '{field}' has {null_count} null values ({null_count / len(df_clean) * 100:.1f}%)"
                )

    # Replace pandas NaT and NaN with None for database compatibility
    df_clean = df_clean.where(pd.notnull(df_clean), None)

    logger.info(
        f"✅ Cleaned data: {len(df_clean)} rows (from {initial_count} initial rows)"
    )
    logger.info(f"📊 Final columns: {list(df_clean.columns)}")

    # Show data quality summary
    logger.info("\n📊 Data Quality Summary:")
    for col in df_clean.columns:
        null_count = df_clean[col].isna().sum()
        null_pct = (null_count / len(df_clean) * 100) if len(df_clean) > 0 else 0
        logger.info(f"  {col}: {null_count} nulls ({null_pct:.1f}%)")

    return df_clean


# ==============================================================================
# ASYNC DATABASE OPERATIONS WITH ERROR HANDLING
# ==============================================================================
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


@retry_on_failure(max_retries=3, delay=2.0)
async def bulk_upsert_data(data: list[dict[str, Any]], batch_size: int = 500):
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
                            if not record.get("registration_number"):
                                logger.warning(
                                    f"Skipping record {i + idx}: missing registration_number"
                                )
                                failed_records.append(record)
                                total_failed += 1
                                continue

                            # Ensure all required fields are present
                            required_fields = [
                                "registration_number",
                                "company_name",
                                "product_name",
                                "brand_name",
                                "type_of_product",
                                "issuance_date",
                                "expiry_date",
                            ]

                            cleaned_record = {}
                            for field in required_fields:
                                cleaned_record[field] = record.get(field)

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
                    stmt = insert(FoodProducts).values(valid_batch)

                    # Update on conflict (if registration_number already exists)
                    update_stmt = stmt.on_conflict_do_update(
                        index_elements=["registration_number"],
                        set_={
                            "company_name": stmt.excluded.company_name,
                            "product_name": stmt.excluded.product_name,
                            "brand_name": stmt.excluded.brand_name,
                            "type_of_product": stmt.excluded.type_of_product,
                            "issuance_date": stmt.excluded.issuance_date,
                            "expiry_date": stmt.excluded.expiry_date,
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
async def verify_insertion(limit: int = 5):
    """Verify data was inserted correctly by querying a few rows"""
    logger.info(f"🔍 Verifying insertion (showing {limit} rows)...")

    try:
        async with async_session() as session:
            result = await session.execute(select(FoodProducts).limit(limit))
            rows = result.scalars().all()

            if rows:
                logger.info(f"✅ Found {len(rows)} rows in database")
                for row in rows:
                    logger.info(f"  - {row.registration_number}: {row.product_name}")
            else:
                logger.warning("⚠️  No rows found in database")
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        logger.debug(traceback.format_exc())
        raise


@retry_on_failure(max_retries=3, delay=1.0)
async def get_record_count() -> int:
    """Get total number of records in database"""
    try:
        async with async_session() as session:
            result = await session.execute(
                select(text("COUNT(*)")).select_from(FoodProducts)
            )
            count = result.scalar()
            return count or 0
    except Exception as e:
        logger.error(f"Error getting record count: {e}")
        return 0


# ==============================================================================
# MAIN EXECUTION WITH ERROR HANDLING
# ==============================================================================
async def process_single_file(file_path: str, use_upsert: bool = True):
    """
    Process a single HTML file and insert into database.
    Includes comprehensive error handling and recovery.

    Args:
        file_path: Path to HTML file
        use_upsert: If True, use upsert (slower but handles duplicates).
    """
    try:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"🚀 Starting processing of: {file_path}")
        logger.info(f"{'=' * 60}\n")

        if not os.path.exists(file_path):  # noqa
            logger.error(f"File not found: {file_path}")
            return

        await create_tables()

        try:
            df = extract_data_from_file(file_path)
        except Exception as e:
            logger.error(f"Failed to extract data: {e}")
            logger.debug(traceback.format_exc())
            return

        if df.empty:
            logger.warning("⚠️  No data extracted, skipping database insertion")
            return

        try:
            df_clean = transform_dataframe(df)
        except Exception as e:
            logger.error(f"Failed to transform data: {e}")
            logger.debug(traceback.format_exc())
            return

        if df_clean.empty:
            logger.warning(
                "⚠️  No valid data after cleaning, skipping database insertion"
            )
            return

        data = df_clean.to_dict("records")

        try:
            await bulk_upsert_data(data, batch_size=500)
        except Exception as e:
            logger.error(f"Failed to insert data: {e}")
            logger.debug(traceback.format_exc())

        try:
            await verify_insertion(limit=5)
        except Exception as e:
            logger.warning(f"Verification failed: {e}")

        try:
            total = await get_record_count()
            logger.info(f"\n📊 Total records in database: {total}")
        except Exception as e:
            logger.warning(f"Could not get record count: {e}")

        logger.info(f"\n✅ Processing complete for {file_path}")

    except Exception as e:
        logger.error(f"\n❌ Critical error processing file: {e}")
        logger.debug(traceback.format_exc())
        raise


async def process_multiple_files(
    folder_path: str, use_upsert: bool = True, file_pattern: str = "*.html"
):
    """
    Process all matching files in a folder and insert into database.
    Continues processing even if individual files fail.

    Args:
        folder_path: Path to folder containing files
        use_upsert: If True, use upsert mode
        file_pattern: File pattern to match (e.g., "*.html", "*.xls")
    """
    try:
        logger.info(f"\n{'=' * 60}")
        logger.info("🚀 Starting batch processing")
        logger.info(f"Folder: {folder_path}")
        logger.info(f"Pattern: {file_pattern}")
        logger.info(f"{'=' * 60}\n")

        if not os.path.exists(folder_path):  # noqa
            logger.error(f"Folder not found: {folder_path}")
            return

        await create_tables()

        path = Path(folder_path)
        files = list(path.glob(file_pattern))

        if file_pattern == "*.html":
            files.extend(list(path.glob("*.xls")))
            files.extend(list(path.glob("*.htm")))
            files.extend(list(path.glob("*.csv")))
        elif file_pattern == "*.*":
            # If a generic pattern is used, include common file types
            files.extend(list(path.glob("*.xls")))
            files.extend(list(path.glob("*.htm")))
            files.extend(list(path.glob("*.csv")))

        if not files:
            logger.warning(
                f"⚠️  No files matching '{file_pattern}' found in {folder_path}"
            )
            return

        logger.info(f"📁 Found {len(files)} files to process")

        all_data = []
        successful_files = 0
        failed_files = []

        for idx, file_path in enumerate(files, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing file {idx}/{len(files)}: {file_path.name}")
            logger.info(f"{'=' * 60}")

            try:
                df = extract_data_from_file(str(file_path))

                if not df.empty:
                    df_clean = transform_dataframe(df)
                    if not df_clean.empty:
                        all_data.extend(df_clean.to_dict("records"))
                        successful_files += 1
                        logger.info(f"✅ Successfully processed {file_path.name}")
                    else:
                        logger.warning(
                            f"⚠️  No valid data after cleaning: {file_path.name}"
                        )
                        failed_files.append(
                            (file_path.name, "No valid data after cleaning")
                        )
                else:
                    logger.warning(f"⚠️  No data extracted: {file_path.name}")
                    failed_files.append((file_path.name, "No data extracted"))

            except Exception as e:
                logger.error(f"❌ Failed to process {file_path.name}: {e}")
                logger.debug(traceback.format_exc())
                failed_files.append((file_path.name, str(e)))
                continue

        logger.info(f"\n{'=' * 60}")
        logger.info("📊 File Processing Summary:")
        logger.info(f"  Total files: {len(files)}")
        logger.info(f"  Successful: {successful_files}")
        logger.info(f"  Failed: {len(failed_files)}")
        if failed_files:
            logger.warning("  Failed files:")
            for filename, reason in failed_files:
                logger.warning(f"    - {filename}: {reason}")
        logger.info(f"{'=' * 60}\n")

        if not all_data:
            logger.warning("⚠️  No data extracted from any files")
            return

        logger.info(f"📊 Total rows extracted from all files: {len(all_data)}")

        try:
            await bulk_upsert_data(all_data, batch_size=500)
        except Exception as e:
            logger.error(f"Failed during bulk insert: {e}")
            logger.debug(traceback.format_exc())

        try:
            await verify_insertion(limit=10)
            total = await get_record_count()
            logger.info(f"\n📊 Total records in database: {total}")
        except Exception as e:
            logger.warning(f"Verification failed: {e}")

    except Exception as e:
        logger.error(f"\n❌ Critical error processing files: {e}")
        logger.debug(traceback.format_exc())
        raise


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    # Configuration

    # Option 1: Process a single file
    SINGLE_FILE_PATH = "data/FoodProduct_Rawmat.xls"

    # Option 2: Process all files in a folder
    FOLDER_PATH = "data/"

    # File pattern to match
    FILE_PATTERN = "*.*"  # This will match all files, with specific extensions added in the processing function

    # Choose processing mode
    USE_UPSERT = True

    logger.info("🚀 Starting FDA Food Products Data Extractor and Database Loader")
    logger.info("=" * 60)

    try:
        # Mode 1: Process single file
        asyncio.run(process_single_file(SINGLE_FILE_PATH, use_upsert=USE_UPSERT))

        # Mode 2: Process all files in folder (uncomment to use)
        # asyncio.run(process_multiple_files(FOLDER_PATH, use_upsert=USE_UPSERT, file_pattern=FILE_PATTERN))

        logger.info("\n✅ All processing complete!")

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Processing interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        logger.debug(traceback.format_exc())
        exit(1)
