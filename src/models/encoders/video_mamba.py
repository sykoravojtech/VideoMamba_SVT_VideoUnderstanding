import torch
from fvcore.common.config import CfgNode

from .encoder_abstract import EncoderAbstract

class VideoMambaEncoder(EncoderAbstract):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.config = config

    def forward(self, X: torch.Tensor):
        return None
