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
 
# Title
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
goal


## Description
desc

## Repository structure
- **assets** (media files for example, display)
- **data**
  - **raw** (original downloaded datasets)
  - **processed** (processed datasets from data/raw)
- **doc** (LaTeX code for the report)
  - **fig** (code for generating images and those images used in the LaTeX report)
- **runs** (experiment log and saved weights)
- **src** (core modules)


## How to run   
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

# This installs cuda & C++ modules causal_conv1d_cuda and selective_scan_cuda. It also downloads 3 model checkpoints to PracticalML_2024/checkpoints/videomamba.
chmod +x ./src/models/encoders/videomamba/setup.sh
./src/models/encoders/videomamba/setup.sh

# training demo on UCF101 dataset
python train.py

# training multi-action classification on Charades dataset
python train.py --config src/config/cls_svt_charades_s224_f8_exp0.yaml
# head-only finetuning
python train_cls_head.py --config src/config/cls_svt_charades_s224_f8_exp0.yaml

# training captioning on Charades dataset
python train.py --config src/config/cap_svt_charades_s224_f8_exp0.yaml
```
Visualize training log at https://wandb.ai/PracticalML2024/PracticalML


## License
Distributed under the MIT License. See `LICENSE` for more information.


### Citation   
```
@article{id,
  title={title},
  author={Vojtěch Sýkora},
  year={2024}
}
```   
