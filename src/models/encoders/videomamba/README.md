# VideoMamba
## State Space Model for Efficient Video Understanding
Taken from https://github.com/OpenGVLab/VideoMamba

## Installing dependencies & checkpoints 
run
```bash
cd PracticalML_2024
chmod +x ./src/models/encoders/videomamba/setup.sh
./src/models/encoders/videomamba/setup.sh
```
This installs cuda & C++ modules `causal_conv1d_cuda` and `selective_scan_cuda`. It also downloads 3 model checkpoints to `PracticalML_2024/checkpoints/videomamba`.

## Checkpoints
https://github.com/OpenGVLab/VideoMamba/blob/main/videomamba/video_sm/MODEL_ZOO.md
- Short-term Video Understanding
    - K400
        - Tiny 16x3x4 https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_t16_k400_f16_res224.pth
        - Small 16x3x4 https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_s16_k400_f16_res224.pth
        - Middle 16x3x4 https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videomamba/videomamba_m16_k400_f16_res224.pth