-- Governed offer-performance KPIs over the impressions fact table: the single
-- semantic source for the Genie Agent (resources/genie.yml). One row of
-- `impressions` = one candidate offer shown to one player at one shop open.
-- {{catalog}} / {{schema}} are job parameters; the unqualified `source` resolves
-- against the USE CATALOG / USE SCHEMA below.

USE CATALOG IDENTIFIER({{catalog}});
USE SCHEMA IDENTIFIER({{schema}});

CREATE OR REPLACE VIEW offer_performance_metrics
WITH METRICS
LANGUAGE YAML
AS $$
version: 1.1
comment: >-
  Harvest Haven shop-offer performance. One source row is one candidate offer shown
  to one player at a shop open (an impression), with whether the player bought it
  and the offer revenue attributed over the following 7 days.
source: impressions

dimensions:
  - name: Event Date
    expr: event_date
    comment: Calendar date of the shop open (UTC).
    synonyms: [date, day, shop open date, impression date]
  - name: Offer ID
    expr: offer_id
    comment: Shop offer identifier, offer_000 to offer_004.
    synonyms: [offer, bundle, SKU]
  - name: Offer Type
    expr: offer_type
    comment: "Offer category: starter, value, whale, seasonal or comeback."
    synonyms: [offer category, bundle type]
  - name: Price Tier
    expr: price_tier
    comment: "Price bucket: tier_1 (~$0.99) up to tier_5 (~$99.99)."
    synonyms: [price point, price band]
  - name: Offer Price USD
    expr: offer_price_usd
    comment: List price of the offer in US dollars.
    synonyms: [price, offer price]
  - name: Country
    expr: country
    comment: Player country as an ISO 3166-1 alpha-2 code (US, FR, JP, ...).
    synonyms: [market, geo, region]
  - name: Player Segment
    expr: segment
    comment: >-
      Spender segment at impression time: whale, dolphin, minnow (payers by spend),
      new (recently installed) or lapsed (returning after a break).
    synonyms: [segment, spender segment, cohort, player type]
  - name: Tenure Band
    expr: CASE
      WHEN tenure_days < 90 THEN '0-89 days'
      WHEN tenure_days < 365 THEN '90-364 days'
      WHEN tenure_days < 730 THEN '1-2 years'
      ELSE '2+ years'
      END
    comment: Days since install, bucketed.
    synonyms: [tenure, player age, account age]

measures:
  - name: Impressions
    expr: COUNT(1)
    comment: Offers shown (one per player, shop open and candidate offer).
    synonyms: [offers shown, views, exposures]
  - name: Purchases
    expr: SUM(purchased)
    comment: Impressions that ended in a purchase of that offer.
    synonyms: [conversions, buys, orders]
  - name: Conversion Rate
    expr: MEASURE(`Purchases`) / MEASURE(`Impressions`)
    comment: Purchases divided by impressions.
    synonyms: [CVR, take rate, purchase rate]
    format:
      type: percentage
      decimal_places: {type: max, places: 1}
  - name: D7 Offer Revenue
    expr: SUM(offer_revenue_d7)
    comment: Offer revenue attributed within 7 days of the impression, in USD.
    synonyms: [revenue, offer revenue, bookings, sales]
    format:
      type: currency
      currency_code: USD
      decimal_places: {type: max, places: 0}
  - name: Revenue per Impression
    expr: MEASURE(`D7 Offer Revenue`) / MEASURE(`Impressions`)
    comment: D7 offer revenue divided by impressions (yield per offer shown).
    synonyms: [RPI, yield, revenue per offer shown]
    format:
      type: currency
      currency_code: USD
      decimal_places: {type: max, places: 2}
  - name: Players Reached
    expr: COUNT(DISTINCT player_id)
    comment: Distinct players who were shown at least one offer.
    synonyms: [reach, players, audience]
  - name: Paying Players
    expr: COUNT(DISTINCT player_id) FILTER (WHERE purchased = 1)
    comment: Distinct players with at least one purchase.
    synonyms: [payers, buyers, converting players]
  - name: ARPPU
    expr: MEASURE(`D7 Offer Revenue`) / MEASURE(`Paying Players`)
    comment: Average D7 offer revenue per paying player.
    synonyms: [average revenue per paying user, revenue per payer]
    format:
      type: currency
      currency_code: USD
      decimal_places: {type: max, places: 2}
$$;
