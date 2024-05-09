import torch

from .encoder_abstract import EncoderAbstract
from .timesformer import get_vit_base_patch16_224

class VideoTransformerEncoder(EncoderAbstract):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.vit = get_vit_base_patch16_224(cfg=config, no_head=True)
        vit = self.vit.embed_dim

    def forward(self, X) -> torch.Tensor:
        return None