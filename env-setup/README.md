### Provisioning A2 based AI Platform Notebook

```

gcloud compute instances create jk-a2-n1 \
   --project jk-mlops-dev \
   --zone us-central1-c \
   --machine-type a2-highgpu-1g \
   --maintenance-policy TERMINATE --restart-on-failure \
   --image-family tf2-ent-2-3-cu110 \
   --image-project deeplearning-platform-release \
   --boot-disk-size 200GB \
   --metadata "install-nvidia-driver=True,proxy-mode=project_editors" \
   --metadata-from-file startup-script=install-dcgm.sh \
   --scopes https://www.googleapis.com/auth/cloud-platform

gcloud notebooks instances register jk-a2-n1 --location us-central1-c

sudo journalctl -u google-startup-scripts.service
   
```
   
```
gcloud notebooks instances create jk-a2-t2 \
--project jk-mlops-dev \
--location us-central1-c \
--vm-image-project deeplearning-platform-release \
--vm-image-family tf2-ent-2-3-cu110 \
--machine-type n1-standard-4 \
--accelerator-core-count 1 \
--accelerator-type NVIDIA_TESLA_V100 \
--metadata "install-nvidia-driver=True,proxy-mode=project_editors" \
--post-startup-script gs://jk-bert-lab-bucket/scripts/install-dcgm.sh
```
