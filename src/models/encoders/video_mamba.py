import sys
import os
# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the videomamba directory
videomamba_path = os.path.join(current_dir, 'videomamba')
# Add the videomamba directory to the system path
sys.path.insert(0, videomamba_path)

import numpy as np
import torch
from fvcore.common.config import CfgNode

from .encoder_abstract import EncoderAbstract
from .videomamba import videomamba_tiny, videomamba_small, videomamba_middle

class VideoMambaEncoder(EncoderAbstract):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.config = config

        np.random.seed(config.SEED)
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        torch.cuda.manual_seed_all(config.SEED)

        kwargs = {
            "pretrained": config.MODEL.ENCODER.PRETRAINED,
            "num_classes": 0, # not config.MODEL.HEAD.NUM_CLASSES because we have our own head
            "num_frames": config.DATA.NUM_SAMPLED_FRAMES,
            "img_size": config.DATA.IMG_SIZE,
            "norm_epsilon": config.TRAIN.OPTIM.EPS,
            "device": "cuda" if config.TRAIN.ACCELERATOR == "auto" else config.TRAIN.ACCELERATOR,
            "return_hidden": config.MODEL.ENCODER.RETURN_ALL_HIDDEN
            # "embed_dim": config.MODEL.ENCODER.HIDDEN_SIZE, # already implemented in the model
        }
        if config.MODEL.ENCODER.MODEL_SIZE == "tiny":
            self.model = videomamba_tiny(**kwargs).cuda()
            # embed_dim = config.MODEL.ENCODER.HIDDEN_SIZE = 192
        elif config.MODEL.ENCODER.MODEL_SIZE == "small":
            self.model = videomamba_small(**kwargs).cuda()
            # embed_dim = config.MODEL.ENCODER.HIDDEN_SIZE = 384
        elif config.MODEL.ENCODER.MODEL_SIZE == "middle":
            self.model = videomamba_middle(**kwargs).cuda()
            # embed_dim = config.MODEL.ENCODER.HIDDEN_SIZE = 576
        else:
            raise ValueError(f"Invalid VideoMamba model size: {config.MODEL.ENCODER.MODEL_SIZE}")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model.forward(X, return_all_hiddens=self.config.MODEL.ENCODER.RETURN_ALL_HIDDEN)

"""
    Check the README.md in the videomamba folder for installing cuda & C++ libraries
    
    Try running: python train.py -c src/config/cls_vm_ucf101_s224_f8_exp0.yaml
"""

"""
ENCODER:
    TYPE: VideoMamba
    MODEL_SIZE: small # tiny, small, middle
    HIDDEN_SIZE: 384  # 192,  384,   576
"""

"""
BUG FIXING
- VideoMamba segmentation fault 
    - tiny / small -> set num_workers > 3
    - middle -> set num_workers < 4
"""