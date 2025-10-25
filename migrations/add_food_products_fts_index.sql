-- Migration: Add functional index to food_products for fast exact ID lookups
-- Description: Adds LOWER() functional index for case-insensitive exact matching
-- Note: food_products already has search_vector and GIN index

-- Create functional index for case-insensitive exact ID lookups
-- This enables O(1) lookups for exact registration number matches
CREATE INDEX IF NOT EXISTS idx_food_products_registration_number_lower
ON food_products (LOWER(registration_number));

-- Optional: Verify existing search_vector and GIN index
-- SELECT indexname, indexdef 
-- FROM pg_indexes 
-- WHERE tablename = 'food_products';

-- Optional: Analyze table to update statistics for query planner
ANALYZE food_products;

-- Verification query (run after migration):
-- SELECT registration_number, brand_name, product_name
-- FROM food_products
-- WHERE LOWER(registration_number) = LOWER('FR-123456')
-- LIMIT 1;
