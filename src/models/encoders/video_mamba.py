import torch

from .encoder_abstract import EncoderAbstract

class VideoMambaEncoder(EncoderAbstract):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
