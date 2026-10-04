# extractor_to_db_food_industry.py
"""Food industry establishment data extractor and database upserter.

Extracts food industry establishment data from HTML tables and upserts it
into the PostgreSQL database with retry logic and error handling.
"""
import asyncio
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

# Import the FoodIndustry model
from app.models.food_industry import FoodIndustry
from app.services.upsert_extractors.common import (
    bulk_upsert,
    clean_string_field,
    extract_data_from_html,
    parse_date_safely,
    process_files,
)
from app.services.upsert_extractors.common import (
    get_record_count as count_records,
)
from app.services.upsert_extractors.common import (
    verify_insertion as verify_records,
)

FIELDS = (
    "license_number",
    "name_of_establishment",
    "owner",
    "address",
    "region",
    "activity",
    "issuance_date",
    "expiry_date",
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
            "license number",
            "license no",
            "license no.",
            "license",
            "lto number",
            "licenseno",
            "lic no",
        ],
        "name_of_establishment": [
            "name of establishment",
            "establishment name",
            "establishment",
            "company name",
            "name",
            "business name",
        ],
        "owner": ["owner", "proprietor", "owner name", "owned by"],
        "address": ["address", "location", "business address", "complete address"],
        "region": ["region", "regional location", "area"],
        "activity": ["activity", "business activity", "type of activity", "business type"],
        "issuance_date": [
            "issuance date",
            "issue date",
            "date of issue",
            "date issued",
            "issued date",
            "issuancedate",
            "issuedate",
            "date of issuance",
        ],
        "expiry_date": [
            "expiry date",
            "expiration date",
            "exp date",
            "valid until",
            "expires",
            "expirydate",
            "expirationdate",
            "validity date",
        ],
    }

    # Create a mapping from actual column names to standardized names
    column_mapping = {}

    for standard_name, possible_names in possible_column_names.items():
        for col in df_clean.columns:
            if isinstance(col, str) and col.casefold() in possible_names:
                column_mapping[col] = standard_name
                break

    logger.info(f"📋 Found column mappings: {column_mapping}")

    # Check for missing critical columns
    required_fields = FIELDS

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
# DATABASE LOADING AND FILE PROCESSING
# ==============================================================================
async def bulk_upsert_data(data: list[dict[str, Any]], batch_size: int = 500):
    """Upsert cleaned rows using the shared food importer loader."""
    await bulk_upsert(FoodIndustry, "license_number", FIELDS, data, batch_size)


async def verify_insertion(limit: int = 5):
    await verify_records(FoodIndustry, "license_number", "name_of_establishment", limit)


async def get_record_count() -> int:
    return await count_records(FoodIndustry)


async def process_single_file(file_path: str):
    """Process a single file and upsert its cleaned rows."""
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {file_path}")
        return
    await process_files(
        [path],
        extract_data_from_html,
        transform_dataframe,
        upsert=bulk_upsert_data,
        verify=verify_insertion,
        count=get_record_count,
    )


async def process_multiple_files(folder_path: str, file_pattern: str = "*.html"):
    """Process matching files, continuing after individual file failures."""
    path = Path(folder_path)
    if not path.exists():
        logger.error(f"Folder not found: {folder_path}")
        return
    files = list(path.glob(file_pattern))
    if file_pattern == "*.html":
        files.extend(path.glob("*.xls"))
        files.extend(path.glob("*.htm"))
    await process_files(
        files,
        extract_data_from_html,
        transform_dataframe,
        upsert=bulk_upsert_data,
        verify=verify_insertion,
        count=get_record_count,
        batch=True,
    )


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

    # --------------------------------------------------------------------

    # Run the processor
    logger.info("🚀 Starting FDA Food Industry Data Extractor and Database Loader")
    logger.info("=" * 60)

    try:
        # Mode 1: Process single file
        asyncio.run(process_single_file(SINGLE_FILE_PATH))

        # Mode 2: Process all files in folder (uncomment to use)
        # asyncio.run(process_multiple_files(FOLDER_PATH, file_pattern=FILE_PATTERN))

        logger.info("\n✅ All processing complete!")

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Processing interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        logger.debug(traceback.format_exc())
        exit(1)
