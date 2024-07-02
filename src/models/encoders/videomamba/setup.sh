#!/bin/bash

### PART 1
# Make sure you are in PracticalML_2024/src/models/encoders/videomamba
# install cuda & C++ dependencies
cd src/models/encoders/videomamba
python setup_causal_conv1d.py install
python setup_selective_scan.py install

### PART 2
# download tiny, small, and middle VideoMamba checkpoints
cd ../../../../ # go to the root directory
# Define the directory and URLs
DIR="checkpoints/videomamba"
URL_TINY="https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_t16_k400_f16_res224.pth"
URL_SMALL="https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_s16_k400_f16_res224.pth"
URL_MIDDLE="https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_k400_f16_res224.pth"

# Create the directory if it doesn't exist
if [ ! -d "$DIR" ]; then
  mkdir -p "$DIR"
  echo "Created directory: $DIR"
else
  echo "Directory already exists: $DIR"
fi

# Change to the directory
cd "$DIR"

# Download the files
wget -O videomamba_t16_k400_f16_res224.pth "$URL_TINY"
wget -O videomamba_s16_k400_f16_res224.pth "$URL_SMALL"
wget -O videomamba_m16_k400_f16_res224.pth "$URL_MIDDLE"

echo "All model checkpoints downloaded to $DIR"