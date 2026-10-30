-- Silver: the offer catalogue. A missing id or non-positive price fails the update,
-- because every downstream join and the model's offer features depend on it.
CREATE OR REFRESH MATERIALIZED VIEW offers (
  CONSTRAINT offer_id_present EXPECT (offer_id IS NOT NULL) ON VIOLATION FAIL UPDATE,
  CONSTRAINT price_positive EXPECT (offer_price_usd > 0) ON VIOLATION FAIL UPDATE
)
COMMENT "The shop's gem offers: type, price tier, price, gem amount and discount."
AS SELECT
  offer_id,
  offer_type,
  price_tier,
  offer_price_usd,
  offer_gem_amount,
  offer_discount_pct
FROM offers_raw;
