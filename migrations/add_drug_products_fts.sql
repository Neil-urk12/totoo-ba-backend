-- Migration: Add Full-Text Search to drug_products table
-- Description: Adds search_vector column and GIN index for fast FTS queries
-- Requirement: PostgreSQL with pg_trgm extension

-- Step 1: Add search_vector column as a generated column
-- This column is automatically maintained by PostgreSQL
ALTER TABLE drug_products 
ADD COLUMN IF NOT EXISTS search_vector tsvector 
GENERATED ALWAYS AS (
    setweight(to_tsvector('english', COALESCE(registration_number, '')), 'A') ||
    setweight(to_tsvector('english', COALESCE(brand_name, '')), 'B') ||
    setweight(to_tsvector('english', COALESCE(generic_name, '')), 'C') ||
    setweight(to_tsvector('english', COALESCE(manufacturer, '')), 'D')
) STORED;

-- Step 2: Create GIN index for fast FTS queries
-- This index dramatically speeds up @@ (full-text search) operations
CREATE INDEX IF NOT EXISTS idx_drug_products_search_vector 
ON drug_products USING GIN (search_vector);

-- Step 3: Create functional index for case-insensitive exact ID lookups
-- This enables O(1) lookups for exact registration number matches
CREATE INDEX IF NOT EXISTS idx_drug_products_registration_number_lower
ON drug_products (LOWER(registration_number));

-- Optional: Analyze table to update statistics for query planner
ANALYZE drug_products;

-- Verification query (run after migration):
-- SELECT registration_number, brand_name, generic_name
-- FROM drug_products
-- WHERE search_vector @@ plainto_tsquery('english', 'biogesic')
-- ORDER BY ts_rank(search_vector, plainto_tsquery('english', 'biogesic')) DESC
-- LIMIT 10;
