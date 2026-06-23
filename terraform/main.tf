# -------------------------
# DATASETS
# -------------------------
resource "google_bigquery_dataset" "raw_tables" {
  dataset_id = "raw_tables"
  location   = "US"
}

resource "google_bigquery_dataset" "staging" {
  dataset_id = "staging"
  location   = "US"
}

resource "google_bigquery_dataset" "marts" {
  dataset_id = "marts"
  location   = "US"
}

resource "google_bigquery_dataset" "intermediate" {
  dataset_id = "staging_intermediate"
  location   = "US"
}
