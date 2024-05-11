import torch

from .encoder_abstract import EncoderAbstract
from .timesformer import get_vit_base_patch16_224

class VideoTransformerEncoder(EncoderAbstract):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.vit = get_vit_base_patch16_224(cfg=config, no_head=True)
        vit = self.vit.embed_dim
        ckpt = torch.load(config.MODEL.ENCODER.PRETRAINED)
        renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}
        msg = self.vit.load_state_dict(renamed_checkpoint, strict=False)
        print(f"Loaded pretrained video transformer: {msg}")

    def forward(self, X) -> torch.Tensor:
        X = X.permute(0, 2, 1, 3, 4) # to B, n_channels, n_frames, h, w
        return self.vit.forward_features(X, get_all=self.config.MODEL.ENCODER.RETURN_ALL_HIDDEN)