-- Enable PostgreSQL trigram extension for fuzzy matching
-- This enables similarity() and other trigram-based functions for handling OCR typos

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create GIN index on brand_name columns for faster fuzzy searches
-- These indexes dramatically speed up similarity() queries

CREATE INDEX IF NOT EXISTS idx_drug_products_brand_name_trgm 
ON drug_products USING gin (brand_name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_food_products_brand_name_trgm 
ON food_products USING gin (brand_name gin_trgm_ops);

-- Optional: Create indexes for generic_name/product_name for future fuzzy matching
CREATE INDEX IF NOT EXISTS idx_drug_products_generic_name_trgm 
ON drug_products USING gin (generic_name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_food_products_product_name_trgm 
ON food_products USING gin (product_name gin_trgm_ops);
