# Creating a Vertex environment

You can use the [Terraform](https://www.terraform.io/) scripts in the `terraform` folder to automatically provision the environment required by the samples. 

The scripts perform the following actions:
1. Enable the required Cloud APIs
2. Create a regional GCS bucket
3. Create an instance of Vertex Notebooks

You can customize your configuration using a set of variables:

|Variable|Required|Default|Description|
|--------|--------|-------|-----------|
|project_id|Yes||A GCP project ID|
|region|Yes||A GCP region for the GCS bucket|
|zone|Yes||A GCP zone for the Notebook instance|
|name_prefix|Yes||A prefix added to the names of provisioned resources|
|machine_type|No|n1-standard-4|A machine type of the  Notebook instance|
|network_name|No|default|A name of the Notebook instance's network|
|subnet_name|No|default|A name of the Notebook instance's subnet|
|boot_disk_size|No|200GB|A size of the Notebook instance's boot disk|
|image_family|No|tf2-ent-latest-gpu|An image family for the Notebook instance|
|gpu_type|No|null|A GPU type of the Notebook instance. By default, the Notebook instance will be provisioned without a GPU|
|gpu_count|No|null|A GPU count of the Notebook instance|
|install_gpu_driver|No|false|Whether to install a GPU driver|
|force_destroy|No|false|Whether to force the removal of the bucket on terraform destroy.**Note that by default the bucket will not be destroyed**.|


To provision the environment:

- Set the `terraform` directory as your current directory

- Update the `terraform/terraform.tfvars` file with the values reflecting your environment. Alternatively you can provide the values using the Terraform CLI `-var` options in the next step

- Execute the following commands. :
```
terraform init
terrafrom apply
```


To destroy the environment, execute:
```
terraform destroy
```
