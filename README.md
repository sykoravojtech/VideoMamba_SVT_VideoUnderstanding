<!--
## Docstrings
"""
This is an example of Google style.

Args:
    param1: This is the first param.
    param2: This is a second param.

Returns:
    This is a description of what is returned.

Raises:
    KeyError: Raises an exception.
"""

https://docs.google.com/document/d/1u-LVvFSsDFmDl7H6Y-cFUUbPc1N2QNrFJSKC9aFDCZs/edit -->

---

<div align="center">    
 
# Video Transformers for Classification and Captioning
</div>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#goal-of-the-project">Goal of the Project</a></li>
    <li><a href="#description">Description</a></li>
    <li><a href="#repository-structure">Repository structure</a></li>
    <li><a href="#how-to-run">How to run </a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#citation">Citation</a></li>
  </ol>
</details>

<!-- <li>
      <a href="#description">Description</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
</li> -->
 
## Goal of the Project
Investigate the application of transformer-based models for video classification and captioning tasks, focusing on the Self-supervised Video Transformer and Video Mamba models.


## Description
In this project we investigated the application of transformer-based models for video classification and captioning tasks, focusing on the Self-supervised Video Transformer and Video Mamba models. We explored the effectiveness of these models on a complex dataset, which features intricate human interactions and daily life activities. Our approach involves adapting the pre-trained SVT encoder for downstream tasks while keeping it frozen, allowing us to assess its performance on out-of-distribution datasets. We also developed a specialized data processing pipeline to handle the unique challenges presented by the dataset. By comparing the performance of SVT with Video Mamba, we provide insights into the capabilities of self-supervised models in video understanding tasks. Our research contributes to the field by demonstrating the potential of transformer-based models in handling complex video data which could potentially reduce reliance on large labeled datasets for effective video analysis in the future.

## Repository structure
- **assets** (media files for example, display)
- **checkpoints** (saved model weights)
- **data** (datasets)
- **src**
    - **config**
    - **datasets**
    - **models**
    - **utils**


## How to run   
### Set-up
```bash
# clone project   
git clone https://github.com/sykoravojtech/PracticalML_2024.git

# install dependencies   
cd PracticalML_2024
pip install -r requirements.txt

# install pytorchvideo
git clone https://github.com/facebookresearch/pytorchvideo
cd pytorchvideo
pip install -v -e .
cd ..

# download pretrained weights
python download_weights.py


# This installs cuda & C++ modules causal_conv1d_cuda and selective_scan_cuda. It also downloads 3 model checkpoints to PracticalML_2024/checkpoints/videomamba.
chmod +x ./src/models/encoders/videomamba/setup.sh
./src/models/encoders/videomamba/setup.sh

```

### Play with the UI app
```bash
streamlit run --server.port 8503 app.py
```

### Models evaluation
```bash
# 1. Evaluation of multi-action classification on Charades dataset (SVT)
python evaluate_cls_model.py --config src/config/cls_svt_charades_s224_f8_exp0.yaml --weight checkpoint
s/cls_svt_charades_s224_f8_exp0/epoch=18-val_mAP=0.165.ckpt

# 2. Evaluation of multi-action classification on Charades dataset (VideoMamba)
python evaluate_cls_model.py --config src/config/cls_vm_charades_s224_f8_exp0.yaml --weight checkpoints/cls_vm_ch_exp7/epoch=142-val_mAP=0.227.ckpt

# 3. Evaluation of captioning on Charades dataset (SVT)
python evaluate_cap_model.py --config src/config/cap_svt_charades_s224_f8_exp0.yaml --weight checkpoints/cap_svt_charades_s224_f8_exp_32_train_all/epoch=11-step=23952.ckpt

# 4. Evaluation of captioning on Charades dataset (VideoMamba)
python evaluate_cap_model.py --config src/config/cap_vm_charades_s224_f8_exp0.yaml --weight checkpoints/cap_vm_charades_s224_f8_exp0_16_train_all/epoch=14-step=29940.ckpt
```

### Training
```bash
# download data
python download_data.py -ucf
python download_data.py -charades
cd data/raw/
mkdir Charades_frames
wget https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_rgb.tar
tar -xvf Charades_v1_rgb.tar
cd ../../..

# generate Charades annotations
python charades_convert_anns.py

# visualize 1 sample of data, the result is saved in assets
python visualize_dataset.py

# visualize 1 sample of Charades multilabel classification dataset
python visualize_dataset.py --config src/config/cls_svt_charades_s224_f8_exp0.yaml
# visualize 1 sample of Charades captioning dataset
python visualize_dataset.py --config src/config/cap_svt_charades_s224_f8_exp0.yaml


# training demo on UCF101 dataset
python train.py

# training multi-action classification on Charades dataset
python train.py --config src/config/cls_svt_charades_s224_f8_exp0.yaml
# head-only finetuning
python create_encoding.py --config src/config/cls_svt_charades_s224_f8_exp0.yaml
python train_cls_head.py --config src/config/cls_svt_charades_s224_f8_exp0.yaml


# training captioning on Charades dataset
python train.py --config src/config/cap_svt_charades_s224_f8_exp0.yaml
# head-only finetuning
python create_encoding.py --config src/config/cls_svt_charades_s224_f8_exp0.yaml
python train_cap_head.py --config src/config/cap_svt_charades_s224_f8_exp0.yaml
```

Visualize training log at https://wandb.ai/PracticalML2024/PracticalML


## License
Distributed under the MIT License. See `LICENSE` for more information.


### Citation   
```
@article{transf_cls_cap,
  title={Video Transformers for Classification and Captioning},
  author={Vojtěch Sýkora, Nam Nguyen The, Leon Trochelmann, Swadesh Jana, Eric Nazarenus},
  year={2024}
}
```   
