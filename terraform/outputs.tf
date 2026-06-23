output "datasets" {
  value = [
    google_bigquery_dataset.raw_tables.dataset_id,
    google_bigquery_dataset.staging.dataset_id,
    google_bigquery_dataset.marts.dataset_id,
    google_bigquery_dataset.staging_intermediate.dataset_id
  ]
}