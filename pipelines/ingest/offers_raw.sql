-- Bronze: Auto Loader ingests the raw offer-catalogue files.
CREATE OR REFRESH STREAMING TABLE offers_raw
COMMENT "Raw offer catalogue exactly as landed, with the source file and ingest time."
AS SELECT
  *,
  _metadata.file_path AS source_file,
  current_timestamp() AS ingested_at
FROM STREAM read_files('${landing_path}/offers', format => 'parquet');
