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

# Check and download the files if they do not exist
if [ ! -f "videomamba_t16_k400_f16_res224.pth" ]; then
  wget -O videomamba_t16_k400_f16_res224.pth "$URL_TINY"
else
  echo "videomamba_t16_k400_f16_res224.pth already exists. Skipping download."
fi

if [ ! -f "videomamba_s16_k400_f16_res224.pth" ]; then
  wget -O videomamba_s16_k400_f16_res224.pth "$URL_SMALL"
else
  echo "videomamba_s16_k400_f16_res224.pth already exists. Skipping download."
fi

if [ ! -f "videomamba_m16_k400_f16_res224.pth" ]; then
  wget -O videomamba_m16_k400_f16_res224.pth "$URL_MIDDLE"
else
  echo "videomamba_m16_k400_f16_res224.pth already exists. Skipping download."
fi

echo "All model checkpoints checked/downloaded to $DIR"
