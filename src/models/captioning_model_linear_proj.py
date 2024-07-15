from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F
from fvcore.common.config import CfgNode
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from .model_abstract import ModelAbstract
from .encoders import create_encoder, EncoderAbstract
from .heads import create_head, HeadAbstract

class VideoCaptioningModelLinear(ModelAbstract):
    def __init__(self, config: CfgNode) -> None:
        super().__init__(config)
        self.visual_mapper = nn.Linear(768, config.MODEL.ENCODER.HIDDEN_SIZE)
        
    def create_encoder(self) -> EncoderAbstract:
        return create_encoder(self.config)

    def create_head(self) -> HeadAbstract:
        return create_head(self.config)

    def forward(self, X: torch.Tensor , y: Dict[str, torch.Tensor]) -> torch.Tensor:
        '''
            Args:
                X: video pixel tensor
                y: dictionary of {'input_ids', 'attention_mask'} for the target caption
            Returns:
                logits: the output logits of the model
        '''
#        enc_hidden = self.encoder(X)
        mapped = self.visual_mapper(X)
        output = self.head(mapped, y)
        return output.logits

    def create_loss_function(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def compute_loss(self, y_pred:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        # Shift so that tokens < n predict n
        labels = y['input_ids']
        shift_logits = y_pred[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            

    def compute_metrics(self, step_outputs) -> Dict[str, float]:
        return dict()

    def generate(self, X: torch.Tensor, max_len: int = 64, beam_size: int = 1) -> str:
        enc_hidden = self.encoder(X)
        mapped = self.visual_mapper(enc_hidden)
        return self.head.beam_search(mapped, max_len, beam_size)
        
    def configure_optimizers(self):
        optimizer = AdamW(list(self.visual_mapper.parameters()) + list(self.head.parameters()), 
                        lr=self.config.TRAIN.OPTIM.INIT_LEARNING_RATE, 
                        eps=self.config.TRAIN.OPTIM.EPS, betas=self.config.TRAIN.OPTIM.BETAS)

        lr_scheduler = MultiStepLR(optimizer, milestones=self.config.TRAIN.OPTIM.LR_MILESTONES, gamma=0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": 'epoch', 
                "frequency": 1, 
            },
        }