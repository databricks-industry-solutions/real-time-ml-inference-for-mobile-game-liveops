-- Bronze: Auto Loader ingests the raw shop-impression files generate_data lands in the
-- project volume (${landing_path} is set in resources/ingest.pipeline.yml).
CREATE OR REFRESH STREAMING TABLE impressions_raw
COMMENT "Raw shop impressions exactly as landed, with the source file and ingest time."
AS SELECT
  *,
  _metadata.file_path AS source_file,
  current_timestamp() AS ingested_at
FROM STREAM read_files('${landing_path}/impressions', format => 'parquet');
