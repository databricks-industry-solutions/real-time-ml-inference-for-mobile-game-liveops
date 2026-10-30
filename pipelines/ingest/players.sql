-- Silver: one row per player, derived from the refined impressions. Player attributes
-- are constant per player in the impression log, so any_value picks the single value.
CREATE OR REFRESH MATERIALIZED VIEW players
COMMENT "One row per player: country, spender segment, tenure and spend history."
AS SELECT
  player_id,
  any_value(country) AS country,
  any_value(segment) AS segment,
  any_value(tenure_days) AS tenure_days,
  any_value(lifetime_spend) AS lifetime_spend,
  any_value(historical_ltv) AS historical_ltv,
  any_value(avg_session_len_min) AS avg_session_len_min,
  any_value(purchases_30d) AS purchases_30d
FROM impressions
GROUP BY player_id;
