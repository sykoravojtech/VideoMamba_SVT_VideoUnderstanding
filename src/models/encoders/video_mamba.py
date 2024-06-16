import sys
# print(f"\n{sys.path=}\n")
sys.path.insert(0, '/teamspace/studios/this_studio/PracticalML_2024/src/models/encoders/videomamba')
# print(f"\n{sys.path=}\n")

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

        if config.MODEL.ENCODER.PRETRAINED:
            pretrained = True

        kwargs = {
            "pretrained": pretrained,
            "num_classes": config.MODEL.HEAD.NUM_CLASSES,
            "img_size": config.DATA.TRAIN_CROP_SIZE,
            "norm_epsilon": config.TRAIN.OPTIM.EPS,
            "device": "cuda" if config.TRAIN.ACCELERATOR == "gpu" else config.TRAIN.ACCELERATOR,
            "embed_dim": config.MODEL.ENCODER.HIDDEN_SIZE,
        }
        if config.MODEL.ENCODER.MODEL_SIZE == "tiny":
            self.model = videomamba_tiny(**kwargs).cuda()
        elif config.MODEL.ENCODER.MODEL_SIZE == "small":
            self.model = videomamba_small(**kwargs).cuda()
        elif config.MODEL.ENCODER.MODEL_SIZE == "middle":
            self.model = videomamba_middle(**kwargs).cuda()
        else:
            raise ValueError(f"Invalid VideoMamba model size: {config.MODEL.ENCODER.MODEL_SIZE}")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model.forward(X)


"""
class VisionMamba(nn.Module):
    def __init__(
            self, 
            img_size=224, 
            patch_size=16, 
            depth=24, 
            embed_dim=192, 
            channels=3, 
            num_classes=400,
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True, 
            residual_in_fp32=True,
            bimamba=True,
            # video
            kernel_size=1, 
            num_frames=16, 
            fc_drop_rate=0., 
            device=None,
            dtype=None,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
        ):
"""