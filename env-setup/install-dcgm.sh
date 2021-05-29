if ! dpkg-query -W datacenter-gpu-manager
then
   echo "Installing NVidia DCGM."
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
   echo "deb http://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list 
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/7fa2af80.pub
   wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-$distribution.pin
   sudo mv cuda-$distribution.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-get update
   sudo apt-get install -y datacenter-gpu-manager
   # Enable DCGM systemd service (on reboot) and start it now
   sudo systemctl --now enable nvidia-dcgm
else
    echo "NVidia DCGM already installed. Skipping installation."
fi
