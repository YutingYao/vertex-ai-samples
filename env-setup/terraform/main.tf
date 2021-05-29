terraform {
  required_version = ">= 0.14"
  required_providers {
    google = "~> 3.6"
  }

  backend "gcs" {
    bucket  = "jk-terraform-state"
    prefix  = "vertex-ai-v-env"
  }
}

provider "google" {
    project   = var.project_id 
}

data "google_project" "project" {}

data "google_compute_default_service_account" "default" {}

output "project" {
  value = data.google_client_config.current.project
}
