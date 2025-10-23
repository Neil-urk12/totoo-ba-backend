-- ============================================================================
-- Full-Text Search (FTS) Migration for FDA Tables
-- ============================================================================
-- This migration adds tsvector columns and GIN indexes for full-text search
-- on all FDA product and industry tables.
--
-- Benefits:
-- - Much faster than ILIKE searches (uses indexed search)
-- - Built-in relevance ranking with ts_rank
-- - Better handling of word variations and stemming
-- - Automatic updates (GENERATED ALWAYS AS)
--
-- To apply: Run this in your Supabase SQL editor
-- ============================================================================

-- ============================================================================
-- 1. FOOD PRODUCTS (Already Applied)
-- ============================================================================
-- Generated tsvector column for unified search
ALTER TABLE food_products
ADD COLUMN IF NOT EXISTS search_vector tsvector
GENERATED ALWAYS AS (
  to_tsvector('english',
    coalesce(registration_number, '') || ' ' ||
    coalesce(company_name, '') || ' ' ||
    coalesce(product_name, '') || ' ' ||
    coalesce(brand_name, '') || ' ' ||
    coalesce(type_of_product, '')
  )
) STORED;

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_food_products_search 
ON food_products USING GIN (search_vector);


-- ============================================================================
-- 2. DRUG PRODUCTS
-- ============================================================================
-- Generated tsvector column for unified search
ALTER TABLE drug_products
ADD COLUMN IF NOT EXISTS search_vector tsvector
GENERATED ALWAYS AS (
  to_tsvector('english',
    coalesce(registration_number, '') || ' ' ||
    coalesce(generic_name, '') || ' ' ||
    coalesce(brand_name, '') || ' ' ||
    coalesce(manufacturer, '') || ' ' ||
    coalesce(dosage_form, '') || ' ' ||
    coalesce(classification, '') || ' ' ||
    coalesce(pharmacologic_category, '')
  )
) STORED;

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_drug_products_search 
ON drug_products USING GIN (search_vector);


-- ============================================================================
-- 3. FOOD INDUSTRY
-- ============================================================================
-- Generated tsvector column for unified search
ALTER TABLE "Food Industry"
ADD COLUMN IF NOT EXISTS search_vector tsvector
GENERATED ALWAYS AS (
  to_tsvector('english',
    coalesce(license_number, '') || ' ' ||
    coalesce(name_of_establishment, '') || ' ' ||
    coalesce(owner, '') || ' ' ||
    coalesce(address, '') || ' ' ||
    coalesce(region, '') || ' ' ||
    coalesce(activity, '')
  )
) STORED;

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_food_industry_search 
ON "Food Industry" USING GIN (search_vector);


-- ============================================================================
-- 4. DRUG INDUSTRY
-- ============================================================================
-- Generated tsvector column for unified search
ALTER TABLE "Drug Industry"
ADD COLUMN IF NOT EXISTS search_vector tsvector
GENERATED ALWAYS AS (
  to_tsvector('english',
    coalesce(license_number, '') || ' ' ||
    coalesce(name_of_establishment, '') || ' ' ||
    coalesce(owner, '') || ' ' ||
    coalesce(address, '') || ' ' ||
    coalesce(region, '') || ' ' ||
    coalesce(activity, '')
  )
) STORED;

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_drug_industry_search 
ON "Drug Industry" USING GIN (search_vector);


-- ============================================================================
-- 5. MEDICAL DEVICE INDUSTRY
-- ============================================================================
-- Generated tsvector column for unified search
ALTER TABLE "Medical Device Industry"
ADD COLUMN IF NOT EXISTS search_vector tsvector
GENERATED ALWAYS AS (
  to_tsvector('english',
    coalesce(license_number, '') || ' ' ||
    coalesce(name_of_establishment, '') || ' ' ||
    coalesce(owner, '') || ' ' ||
    coalesce(address, '') || ' ' ||
    coalesce(region, '') || ' ' ||
    coalesce(activity, '')
  )
) STORED;

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_medical_device_industry_search 
ON "Medical Device Industry" USING GIN (search_vector);


-- ============================================================================
-- 6. COSMETIC INDUSTRY
-- ============================================================================
-- Generated tsvector column for unified search
ALTER TABLE "Cosmetic Industry"
ADD COLUMN IF NOT EXISTS search_vector tsvector
GENERATED ALWAYS AS (
  to_tsvector('english',
    coalesce(license_number, '') || ' ' ||
    coalesce(name_of_establishment, '') || ' ' ||
    coalesce(owner, '') || ' ' ||
    coalesce(address, '') || ' ' ||
    coalesce(region, '') || ' ' ||
    coalesce(activity, '')
  )
) STORED;

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_cosmetic_industry_search 
ON "Cosmetic Industry" USING GIN (search_vector);


-- ============================================================================
-- 7. DRUGS NEW APPLICATIONS
-- ============================================================================
-- Generated tsvector column for unified search
ALTER TABLE "Drugs New Applications"
ADD COLUMN IF NOT EXISTS search_vector tsvector
GENERATED ALWAYS AS (
  to_tsvector('english',
    coalesce(document_tracking_number, '') || ' ' ||
    coalesce(applicant_company, '') || ' ' ||
    coalesce(brand_name, '') || ' ' ||
    coalesce(generic_name, '') || ' ' ||
    coalesce(dosage_form, '') || ' ' ||
    coalesce(pharmacologic_category, '') || ' ' ||
    coalesce(application_type, '')
  )
) STORED;

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_drug_applications_search 
ON "Drugs New Applications" USING GIN (search_vector);


-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================
-- Run these to verify the indexes were created successfully:

-- Check all search_vector columns
SELECT 
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE column_name = 'search_vector'
ORDER BY table_name;

-- Check all GIN indexes
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE indexdef LIKE '%GIN%search_vector%'
ORDER BY tablename;

-- Test a sample FTS query on food_products
SELECT 
    registration_number,
    brand_name,
    product_name,
    ts_rank(search_vector, plainto_tsquery('english', 'nestle chocolate')) as rank
FROM food_products
WHERE search_vector @@ plainto_tsquery('english', 'nestle chocolate')
ORDER BY rank DESC
LIMIT 10;


-- ============================================================================
-- PERFORMANCE NOTES
-- ============================================================================
-- 1. The GIN indexes will make searches MUCH faster (typically 10-100x)
-- 2. Initial index creation may take some time for large tables
-- 3. Indexes are automatically maintained by PostgreSQL
-- 4. search_vector columns are automatically updated (GENERATED ALWAYS AS)
-- 5. Use plainto_tsquery() for natural language queries
-- 6. Use to_tsquery() for more advanced boolean queries (AND, OR, NOT)
--
-- Example FTS query performance comparison:
-- - ILIKE search: ~500ms for 100k rows
-- - FTS with GIN: ~5ms for 100k rows
-- ============================================================================
