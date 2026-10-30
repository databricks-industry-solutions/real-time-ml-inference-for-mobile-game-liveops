-- Silver: the refined impression log that setup_stream, preprocess, publish_features
-- and the offer_performance_metrics metric view read. Rows that break the purchase
-- or revenue contract are dropped and counted in the pipeline event log.
CREATE OR REFRESH STREAMING TABLE impressions (
  CONSTRAINT player_present EXPECT (player_id IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT offer_present EXPECT (offer_id IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT event_time_present EXPECT (event_ts IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT purchase_is_flag EXPECT (purchased IN (0, 1)) ON VIOLATION DROP ROW,
  CONSTRAINT revenue_not_negative EXPECT (offer_revenue_d7 >= 0) ON VIOLATION DROP ROW,
  CONSTRAINT revenue_only_on_purchase
    EXPECT (purchased = 1 OR offer_revenue_d7 = 0) ON VIOLATION DROP ROW
)
COMMENT "One row per player, shop open and candidate offer: player attributes at the time, the offer shown, whether it was bought and its 7-day revenue."
CLUSTER BY (event_date)
AS SELECT
  player_id,
  country,
  segment,
  tenure_days,
  lifetime_spend,
  historical_ltv,
  avg_session_len_min,
  purchases_30d,
  offer_id,
  offer_type,
  price_tier,
  offer_price_usd,
  offer_gem_amount,
  offer_discount_pct,
  gem_balance,
  minutes_since_last_purchase,
  match_count,
  event_ts,
  purchased,
  offer_revenue_d7,
  to_date(event_ts) AS event_date
FROM STREAM(impressions_raw);
