variable "project_id" {
    description = "The GCP project ID"
    type        = string
}

variable "region" {
    description = "The region for the environment's components"
    type        = string
}

variable "zone" {
    description = "The zone for the GCE node"
    type        = string
}

variable "name_prefix" {
    description = "The name prefix to add to the resource names"
    type        = string
}

variable "machine_type" {
    description = "The node's machine type"
    type        = string
}

variable "network_name" {
  description = "The network name"
  type        = string
  default     = "default"
}

variable "subnet_name" {
  description = "The subnet name"
  type        = string
  default     = "default"
}

variable "boot_disk_size" {
    description = "The size of the boot disk"
    default     = 100
}

variable "image_family" {
    description = "The Deep Learning image family"
    type        = string
    default     = "common-cu110"
}


variable "gpu_type" {
    description = "GPU type"
    type        = string
    default     = null
}

variable "gpu_count" {
    description = "GPU count"
    type        = string
    default     = null
}

variable "install_gpu_driver" {
    description = "Install GPU driver"
    type        = bool
    default     = false
}