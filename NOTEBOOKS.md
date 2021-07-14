
# Cloud Notebooks

- REST API reference: https://cloud.google.com/notebooks/docs/reference/rest/v1/projects.locations.instances
- gcloud reference: https://cloud.google.com/notebooks/docs/create-new#create-command-line

## REST API examples

### List notebook instances

```
export LOCATION=us-central1
export PROJECT=jk-vertex-demos


curl -X GET \
-H "Authorization: Bearer "$(gcloud auth application-default print-access-token) \
-H "Content-Type: application/json; charset=utf-8" \
https://notebooks.googleapis.com/v1/projects/$PROJECT/locations/$LOCATION/instances

```

### Create a notebook instance 

#### Create a request body

The `instance_owners` field determines who can access the instance. Format: alias@example.com. Currently supports one owner only. If not specified, all of the service account users of your VM instance's service account can use the instance. 

```
cat << EOF > request.json
{
    "name": "jk-notebook-1",
    "machine_type": "n1-standard-4",
    "instance_owners": [
         "user@your-org.com"
    ],
    "vm_image": {
        "project": "deeplearning-platform-release",
        "image_family": "tf2-ent-latest-gpu"
    }

}
EOF
```

#### Submit a request

```
INSTANCE_ID=jk-notebook-1

curl -X POST \
-H "Authorization: Bearer "$(gcloud auth application-default print-access-token) \
-H "Content-Type: application/json; charset=utf-8" \
-d @request.json \
https://notebooks.googleapis.com/v1/projects/$PROJECT/locations/$LOCATION/instances?instanceId=$INSTANCE_ID

```
## gcloud examples

### Creating a notebook instance 

```
export INSTANCE_NAME="jk-example-instance"
export VM_IMAGE_PROJECT="deeplearning-platform-release"
export VM_IMAGE_FAMILY="tf2-2-3-cpu"
export MACHINE_TYPE="n1-standard-4"
export LOCATION="us-central1-b"

gcloud notebooks instances create $INSTANCE_NAME \
  --vm-image-project=$VM_IMAGE_PROJECT \
  --vm-image-family=$VM_IMAGE_FAMILY \
  --machine-type=$MACHINE_TYPE --location=$LOCATION
```
