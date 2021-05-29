resource "google_storage_bucket" "artifact_repo" {
  name          = "${var.name_prefix}-bucket"
  location      = var.region
  storage_class = "REGIONAL"
  force_destroy = false
}