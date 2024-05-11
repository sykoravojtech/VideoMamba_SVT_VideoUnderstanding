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
- **constants** (include constants that are used in multiple files)
- **data**
  - **raw** (original downloaded datasets)
  - **final** (processed datasets from data/raw)
- **doc** (LaTeX code for the report)
  - **fig** (code for generating images and those images used in the LaTeX report)
- **exp** (experiments)
- **scripts** (general data preprocessing code)
- **src** (general functions)


## How to run   
First, install dependencies   
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
rm -r pytorchvideo

# download pretrained weights
python download_weights.py

# download UFC101 data (video classification)
python download_data.py

# visualize 1 sample of data, the result is saved in assets
python visualize_dataset.py

# training
python train.py
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