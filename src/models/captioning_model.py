from typing import Dict, List

from torch import nn
from fvcore.common.config import CfgNode

from .model_abstract import ModelAbstract
from .encoders import create_encoder, EncoderAbstract
from .heads import create_head, HeadAbstract

class VideoCaptioningnModel(ModelAbstract):
    def __init__(self, config: CfgNode) -> None:
        super().__init__(config)

    def create_encoder(self) -> EncoderAbstract:
        return create_encoder(self.config)

    def create_head(self) -> HeadAbstract:
        return create_head(self.config)

    def create_loss_function(self) -> nn.Module:
        return nn.CrossEntropyLoss()
            
    def compute_metrics(self, step_outputs: List) -> Dict:
        
        return {'ROUGE': None}
