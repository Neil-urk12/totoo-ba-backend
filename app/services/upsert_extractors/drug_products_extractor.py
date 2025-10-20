# extractor_to_db.py
import asyncio
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

# Import your database configuration
from app.core.database import Base, async_session, engine

# Import the DrugProducts model
from app.models.drug_products import DrugProducts

load_dotenv()


# ==============================================================================
# EXTRACT DATA FROM HTML TABLE
# ==============================================================================
def extract_data_from_html(file_path: str) -> pd.DataFrame:
    """
    Extract all rows from HTML table and return as pandas DataFrame.

    Args:
        file_path: Path to the HTML file containing the table

    Returns:
        DataFrame with extracted data
    """

    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    try:
        with file.open(encoding="utf-8") as f:
            content = f.read()

        soup = BeautifulSoup(content, "html.parser")
        tables = soup.find_all("table")

        if not tables:
            return pd.DataFrame()


        # Process the first table (modify if you need multiple tables)
        table = tables[0]

        # Extract all rows
        rows = table.find_all("tr")

        if len(rows) < 2:
            return pd.DataFrame()

        # Extract headers from first row
        header_row = rows[0]
        headers = [
            cell.get_text(strip=True) for cell in header_row.find_all(["th", "td"])
        ]


        # Extract data from remaining rows
        data = []
        for row in rows[1:]:  # Skip header row
            cells = row.find_all(["td", "th"])
            row_data = [cell.get_text(strip=True) for cell in cells]

            # Only add rows that have data
            if row_data and any(cell for cell in row_data):
                data.append(row_data)

        # Create DataFrame
        return pd.DataFrame(data, columns=headers)



    except Exception:
        raise


# ==============================================================================
# DATA TRANSFORMATION & CLEANING
# ==============================================================================
def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform the extracted data to match DrugProducts database schema.
    Adjust column mappings based on your actual HTML table headers.

    Args:
        df: Raw DataFrame from HTML extraction

    Returns:
        Cleaned DataFrame ready for database insertion
    """

    # Make a copy to avoid modifying original
    df_clean = df.copy()

    # Define possible column names for each field (case-insensitive)
    possible_column_names = {
        "registration_number": [
            "Registration Number",
            "Registration No",
            "Registration No.",
            "Registration",
            "Reg. No.",
            "registration number",
            "registration no",
            "registration",
            "reg number",
            "reg no",
            "reg no.",
            "regnumber",
            "regno",
            "REGISTRATION NUMBER",
            "REGISTRATION NO",
            "REGISTRATION",
            "REG NO",
        ],
        "product_information": [
            "Product Information",
            "Product Info",
            "Product Details",
            "Product Description",
            "product information",
            "product info",
            "product details",
            "product description",
            "PRODUCT INFORMATION",
            "PRODUCT INFO",
            "PRODUCT DETAILS",
            "PRODUCT DESCRIPTION",
        ],
        "generic_name": [
            "Generic Name",
            "Active Ingredient",
            "Active Substance",
            "generic name",
            "active ingredient",
            "active substance",
            "GENERIC NAME",
            "ACTIVE INGREDIENT",
            "ACTIVE SUBSTANCE",
        ],
        "brand_name": [
            "Brand Name",
            "Trade Name",
            "Product Name",
            "brand name",
            "trade name",
            "product name",
            "BRAND NAME",
            "TRADE NAME",
            "PRODUCT NAME",
        ],
        "dosage_strength": [
            "Dosage Strength",
            "Strength",
            "Concentration",
            "dosage strength",
            "strength",
            "concentration",
            "DOSAGE STRENGTH",
            "STRENGTH",
            "CONCENTRATION",
        ],
        "dosage_form": [
            "Dosage Form",
            "Form",
            "Formulation",
            "dosage form",
            "form",
            "formulation",
            "DOSAGE FORM",
            "FORM",
            "FORMULATION",
        ],
        "classification": [
            "Classification",
            "Class",
            "Type",
            "classification",
            "class",
            "type",
            "CLASSIFICATION",
            "CLASS",
            "TYPE",
        ],
        "packaging": [
            "Packaging",
            "Package",
            "Pack Size",
            "packaging",
            "package",
            "pack size",
            "PACKAGING",
            "PACKAGE",
            "PACK SIZE",
        ],
        "pharmacologic_category": [
            "Pharmacologic Category",
            "Therapeutic Category",
            "Category",
            "pharmacologic category",
            "therapeutic category",
            "category",
            "PHARMACOLOGIC CATEGORY",
            "THERAPEUTIC CATEGORY",
            "CATEGORY",
        ],
        "manufacturer": [
            "Manufacturer",
            "Producer",
            "Manufacturing Company",
            "manufacturer",
            "producer",
            "manufacturing company",
            "MANUFACTURER",
            "PRODUCER",
            "MANUFACTURING COMPANY",
        ],
        "country_of_origin": [
            "Country of Origin",
            "Origin",
            "Manufacturing Country",
            "country of origin",
            "origin",
            "manufacturing country",
            "COUNTRY OF ORIGIN",
            "ORIGIN",
            "MANUFACTURING COUNTRY",
        ],
        "trader": ["Trader", "Distributor Agent", "trader", "TRADER"],
        "importer": ["Importer", "Import Company", "importer", "IMPORTER"],
        "distributor": [
            "Distributor",
            "Distribution Company",
            "distributor",
            "DISTRIBUTOR",
        ],
        "application_type": [
            "Application Type",
            "App Type",
            "application type",
            "app type",
            "APPLICATION TYPE",
            "APP TYPE",
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
        ],
    }

    # Create a mapping from actual column names to standardized names
    column_mapping = {}

    for standard_name, possible_names in possible_column_names.items():
        for col in df_clean.columns:
            if col in possible_names:
                column_mapping[col] = standard_name
                break  # Only map the first match for each standard name

    # Print what mappings were found

    # Rename columns based on mapping
    df_clean = df_clean.rename(columns=column_mapping)

    # Handle date columns (using the standard names)
    date_columns = ["issuance_date", "expiry_date"]
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")

    # Clean string columns - remove extra whitespace
    all_possible_string_columns = [
        "registration_number",
        "product_information",
        "generic_name",
        "brand_name",
        "dosage_strength",
        "dosage_form",
        "classification",
        "packaging",
        "pharmacologic_category",
        "manufacturer",
        "country_of_origin",
        "trader",
        "importer",
        "distributor",
        "application_type",
    ]

    for col in all_possible_string_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].str.strip()

    # Remove rows with missing critical data (using the mapped column name)
    # Try to find the appropriate column name to use
    reg_col = None
    for col in df_clean.columns:
        if col == "registration_number":
            reg_col = col
            break

    if reg_col:
        df_clean = df_clean.dropna(subset=[reg_col])

    # Replace NaN with None for database compatibility
    return df_clean.where(pd.notnull(df_clean), None)




# ==============================================================================
# ASYNC DATABASE OPERATIONS
# ==============================================================================
async def create_tables():
    """Create database tables if they don't exist"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def bulk_upsert_data(data: list[dict[str, Any]], batch_size: int = 1000):
    """
    Insert data into database with conflict resolution (upsert).
    Uses PostgreSQL's ON CONFLICT clause for efficient upserts.

    Args:
        data: List of dictionaries containing row data
        batch_size: Number of rows to insert per batch
    """

    async with async_session() as session:
        try:
            # Process in batches for better performance

            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]

                # PostgreSQL INSERT ... ON CONFLICT (upsert)
                stmt = insert(DrugProducts).values(batch)

                # Update on conflict (if registration_number already exists)
                update_stmt = stmt.on_conflict_do_update(
                    index_elements=["registration_number"],
                    set_={
                        "product_information": stmt.excluded.product_information,
                        "generic_name": stmt.excluded.generic_name,
                        "brand_name": stmt.excluded.brand_name,
                        "dosage_strength": stmt.excluded.dosage_strength,
                        "dosage_form": stmt.excluded.dosage_form,
                        "classification": stmt.excluded.classification,
                        "packaging": stmt.excluded.packaging,
                        "pharmacologic_category": stmt.excluded.pharmacologic_category,
                        "manufacturer": stmt.excluded.manufacturer,
                        "country_of_origin": stmt.excluded.country_of_origin,
                        "trader": stmt.excluded.trader,
                        "importer": stmt.excluded.importer,
                        "distributor": stmt.excluded.distributor,
                        "application_type": stmt.excluded.application_type,
                        "issuance_date": stmt.excluded.issuance_date,
                        "expiry_date": stmt.excluded.expiry_date,
                    },
                )

                await session.execute(update_stmt)
                await session.commit()

                (i // batch_size) + 1
                (len(data) + batch_size - 1) // batch_size


        except Exception:
            await session.rollback()
            raise


async def bulk_insert_simple(data: list[dict[str, Any]], batch_size: int = 1000):
    """
    Simple bulk insert without conflict resolution (faster but fails on duplicates).
    Use this if your data has no duplicates.

    Args:
        data: List of dictionaries containing row data
        batch_size: Number of rows to insert per batch
    """

    async with async_session() as session:
        try:
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]

                # Direct bulk insert using Core
                stmt = insert(DrugProducts).values(batch)
                await session.execute(stmt)
                await session.commit()

                (i // batch_size) + 1
                (len(data) + batch_size - 1) // batch_size


        except Exception:
            await session.rollback()
            raise


async def verify_insertion(limit: int = 5):
    """Verify data was inserted correctly by querying a few rows"""

    async with async_session() as session:
        result = await session.execute(select(DrugProducts).limit(limit))
        rows = result.scalars().all()

        if rows:
            for _row in rows:
                pass
        else:
            pass


async def get_record_count() -> int:
    """Get total number of records in database"""
    async with async_session() as session:
        result = await session.execute(
            select(text("COUNT(*)")).select_from(DrugProducts)
        )
        count = result.scalar()
        return count or 0


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
async def process_single_file(file_path: str, use_upsert: bool = True):
    """
    Process a single HTML file and insert into database.

    Args:
        file_path: Path to HTML file
        use_upsert: If True, use upsert (slower but handles duplicates).
                   If False, use simple insert (faster but fails on duplicates)
    """
    try:
        # Step 1: Create tables
        await create_tables()

        # Step 2: Extract data from HTML
        df = extract_data_from_html(file_path)

        if df.empty:
            return

        # Step 3: Transform data
        df_clean = transform_dataframe(df)

        # Step 4: Convert to list of dictionaries
        data = df_clean.to_dict("records")

        # Step 5: Insert into database
        if use_upsert:
            await bulk_upsert_data(data, batch_size=1000)
        else:
            await bulk_insert_simple(data, batch_size=1000)

        # Step 6: Verify insertion
        await verify_insertion(limit=5)

        # Step 7: Show total count
        await get_record_count()

    except Exception:
        raise


async def process_multiple_files(folder_path: str, use_upsert: bool = True):
    """
    Process all HTML/XLS files in a folder and insert into database.

    Args:
        folder_path: Path to folder containing HTML files
        use_upsert: If True, use upsert mode
    """
    try:
        # Create tables once
        await create_tables()

        # Find all HTML files
        folder = Path(folder_path)
        files = [
            f for f in folder.iterdir() if f.suffix in (".html", ".xls", ".htm")
        ]

        if not files:
            return


        all_data = []

        # Process each file
        for _idx, file in enumerate(files, 1):
            file_path = str(file)  # Convert Path to string

            try:
                # Extract and transform
                df = extract_data_from_html(file_path)
                if not df.empty:
                    df_clean = transform_dataframe(df)
                    all_data.extend(df_clean.to_dict("records"))
            except Exception:
                continue

        if not all_data:
            return


        # Bulk insert all data
        if use_upsert:
            await bulk_upsert_data(all_data, batch_size=1000)
        else:
            await bulk_insert_simple(all_data, batch_size=1000)

        # Verify
        await verify_insertion(limit=10)
        await get_record_count()

    except Exception:
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
    FOLDER_PATH = ""

    # Choose processing mode
    USE_UPSERT = True  # True = handle duplicates, False = handle duplicates, False = faster but fails on duplicates

    # --------------------------------------------------------------------

    # Run the processor

    # Mode 1: Process single file
    asyncio.run(process_single_file(SINGLE_FILE_PATH, use_upsert=USE_UPSERT))

    # Mode 2: Process all files in folder (recommended for multiple files)
    # asyncio.run(process_multiple_files(FOLDER_PATH, use_upsert=USE_UPSERT))

