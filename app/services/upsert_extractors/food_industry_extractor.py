# extractor_to_db_food_industry.py
"""Food industry establishment data extractor and database upserter.

Extracts food industry establishment data from HTML tables and upserts it
into the PostgreSQL database with retry logic and error handling.
"""
import asyncio
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError

# Import your database configuration
from app.core.database import Base, async_session, engine

# Import the FoodIndustry model
from app.models.food_industry import FoodIndustry
from app.services.upsert_extractors.common import (
    clean_string_field,
    extract_data_from_html,
    get_session_with_rollback,
    parse_date_safely,
    retry_on_failure,
)

load_dotenv()


# ==============================================================================
# DATA TRANSFORMATION & CLEANING
# ==============================================================================
def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform the extracted data to match FoodIndustry database schema.
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

    # Make a copy to avoid modifying original
    df_clean = df.copy()

    # Define possible column names for each field (case-insensitive)
    possible_column_names = {
        "license_number": [
            "License Number",
            "License No",
            "License No.",
            "License",
            "LTO Number",
            "license number",
            "license no",
            "license",
            "lto number",
            "licenseno",
            "LICENSE NUMBER",
            "LICENSE NO",
            "LICENSE",
            "LTO NUMBER",
            "Lic No",
            "lic no",
        ],
        "name_of_establishment": [
            "Name of Establishment",
            "Establishment Name",
            "Establishment",
            "Company Name",
            "name of establishment",
            "establishment name",
            "establishment",
            "company name",
            "NAME OF ESTABLISHMENT",
            "ESTABLISHMENT NAME",
            "ESTABLISHMENT",
            "COMPANY NAME",
            "Name",
            "name",
            "NAME",
            "Business Name",
            "business name",
        ],
        "owner": [
            "Owner",
            "Proprietor",
            "Owner Name",
            "Owned By",
            "owner",
            "proprietor",
            "owner name",
            "owned by",
            "OWNER",
            "PROPRIETOR",
            "OWNER NAME",
            "OWNED BY",
        ],
        "address": [
            "Address",
            "Location",
            "Business Address",
            "Complete Address",
            "address",
            "location",
            "business address",
            "complete address",
            "ADDRESS",
            "LOCATION",
            "BUSINESS ADDRESS",
            "COMPLETE ADDRESS",
        ],
        "region": [
            "Region",
            "Regional Location",
            "Area",
            "region",
            "regional location",
            "area",
            "REGION",
            "REGIONAL LOCATION",
            "AREA",
        ],
        "activity": [
            "Activity",
            "Business Activity",
            "Type of Activity",
            "Business Type",
            "activity",
            "business activity",
            "type of activity",
            "business type",
            "ACTIVITY",
            "BUSINESS ACTIVITY",
            "TYPE OF ACTIVITY",
            "BUSINESS TYPE",
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
        "license_number",
        "name_of_establishment",
        "owner",
        "address",
        "region",
        "activity",
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
        # License Number - required, must be unique
        if "license_number" in df_clean.columns:
            df_clean["license_number"] = df_clean["license_number"].apply(
                lambda x: clean_string_field(x, max_length=100)
            )

            sample = df_clean["license_number"].head(10).tolist()
            logger.info(f"Sample cleaned license numbers: {sample}")

        # Name of Establishment - required
        if "name_of_establishment" in df_clean.columns:
            df_clean["name_of_establishment"] = df_clean["name_of_establishment"].apply(
                lambda x: clean_string_field(x, max_length=500)
            )

        # Owner - required
        if "owner" in df_clean.columns:
            df_clean["owner"] = df_clean["owner"].apply(
                lambda x: clean_string_field(x, max_length=300)
            )

        # Address - required
        if "address" in df_clean.columns:
            df_clean["address"] = df_clean["address"].apply(
                lambda x: clean_string_field(x, max_length=500)
            )

        # Region - required
        if "region" in df_clean.columns:
            df_clean["region"] = df_clean["region"].apply(
                lambda x: clean_string_field(x, max_length=100)
            )

        # Activity - required
        if "activity" in df_clean.columns:
            df_clean["activity"] = df_clean["activity"].apply(
                lambda x: clean_string_field(x, max_length=300)
            )

        # Date columns - required
        for date_col in ["issuance_date", "expiry_date"]:
            if date_col in df_clean.columns:
                df_clean[date_col] = df_clean[date_col].apply(parse_date_safely)

    except Exception as e:
        logger.error(f"Error during field cleaning: {e}")
        logger.debug(traceback.format_exc())
        raise

    # Data quality checks and cleaning
    initial_count = len(df_clean)

    # Remove rows with missing license number (primary key)
    if "license_number" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["license_number"])
        logger.info(
            f"Removed {initial_count - len(df_clean)} rows with missing license_number"
        )

    # Remove duplicate license numbers, keep first occurrence
    duplicates = df_clean.duplicated(subset=["license_number"], keep="first")
    if duplicates.any():
        num_duplicates = duplicates.sum()
        logger.warning(
            f"Found {num_duplicates} duplicate license numbers, keeping first occurrence"
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
            # Log a few examples
            invalid_samples = df_clean[invalid_dates][
                ["license_number", "issuance_date", "expiry_date"]
            ].head()
            logger.warning(f"Examples:\n{invalid_samples}")

    # Check for required fields with None values
    required_not_null = [
        "name_of_establishment",
        "owner",
        "address",
        "region",
        "activity",
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
                            if not record.get("license_number"):
                                logger.warning(
                                    f"Skipping record {i + idx}: missing license_number"
                                )
                                failed_records.append(record)
                                total_failed += 1
                                continue

                            # Ensure all required fields are present
                            required_fields = [
                                "license_number",
                                "name_of_establishment",
                                "owner",
                                "address",
                                "region",
                                "activity",
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
                    stmt = insert(FoodIndustry).values(valid_batch)

                    # Update on conflict (if license_number already exists)
                    update_stmt = stmt.on_conflict_do_update(
                        index_elements=["license_number"],
                        set_={
                            "name_of_establishment": stmt.excluded.name_of_establishment,
                            "owner": stmt.excluded.owner,
                            "address": stmt.excluded.address,
                            "region": stmt.excluded.region,
                            "activity": stmt.excluded.activity,
                            "issuance_date": stmt.excluded.issuance_date,
                            "expiry_date": stmt.excluded.expiry_date,
                        },
                    )

                    await session.execute(update_stmt)
                    await session.flush()  # Flush to catch any errors before commit

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
            result = await session.execute(select(FoodIndustry).limit(limit))
            rows = result.scalars().all()

            if rows:
                logger.info(f"✅ Found {len(rows)} rows in database")
                for row in rows:
                    logger.info(
                        f"  - {row.license_number}: {row.name_of_establishment}"
                    )
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
                select(text("COUNT(*)")).select_from(FoodIndustry)
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

        # Step 1: Validate file exists
        if not os.path.exists(file_path):  # noqa
            logger.error(f"File not found: {file_path}")
            return

        # Step 2: Create tables
        await create_tables()

        # Step 3: Extract data from HTML
        try:
            df = extract_data_from_html(file_path)
        except Exception as e:
            logger.error(f"Failed to extract data: {e}")
            logger.debug(traceback.format_exc())
            return

        if df.empty:
            logger.warning("⚠️  No data extracted, skipping database insertion")
            return

        # Step 4: Transform data
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

        # Step 5: Convert to list of dictionaries
        data = df_clean.to_dict("records")

        # Step 6: Insert into database
        try:
            await bulk_upsert_data(data, batch_size=500)
        except Exception as e:
            logger.error(f"Failed to insert data: {e}")
            logger.debug(traceback.format_exc())
            # Continue to verification anyway

        # Step 7: Verify insertion
        try:
            await verify_insertion(limit=5)
        except Exception as e:
            logger.warning(f"Verification failed: {e}")

        # Step 8: Show total count
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

        # Validate folder exists
        if not os.path.exists(folder_path):  # noqa
            logger.error(f"Folder not found: {folder_path}")
            return

        # Create tables once
        await create_tables()

        # Find all matching files
        path = Path(folder_path)
        files = list(path.glob(file_pattern))

        # Also try .xls extension
        if file_pattern == "*.html":
            files.extend(list(path.glob("*.xls")))
            files.extend(list(path.glob("*.htm")))

        if not files:
            logger.warning(
                f"⚠️  No files matching '{file_pattern}' found in {folder_path}"
            )
            return

        logger.info(f"📁 Found {len(files)} files to process")

        all_data = []
        successful_files = 0
        failed_files = []

        # Process each file
        for idx, file_path in enumerate(files, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing file {idx}/{len(files)}: {file_path.name}")
            logger.info(f"{'=' * 60}")

            try:
                # Extract and transform
                df = extract_data_from_html(str(file_path))

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

        # Summary of file processing
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

        # Bulk insert all data
        try:
            await bulk_upsert_data(all_data, batch_size=500)
        except Exception as e:
            logger.error(f"Failed during bulk insert: {e}")
            logger.debug(traceback.format_exc())

        # Verify
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
    # --------------------------------------------------------------------

    # Option 1: Process a single file
    SINGLE_FILE_PATH = ""

    # Option 2: Process all files in a folder
    FOLDER_PATH = "data/"

    # File pattern to match
    FILE_PATTERN = "*.html"  # or "*.xls", "*.htm", etc.

    # Choose processing mode
    USE_UPSERT = (
        True  # True = handle duplicates, False = faster but fails on duplicates
    )

    # --------------------------------------------------------------------

    # Run the processor
    logger.info("🚀 Starting FDA Food Industry Data Extractor and Database Loader")
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
