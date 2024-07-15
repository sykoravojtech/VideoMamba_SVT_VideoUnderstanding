from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F
from fvcore.common.config import CfgNode

from .model_abstract import ModelAbstract
from .encoders import create_encoder, EncoderAbstract
from .heads import create_head, HeadAbstract

class VideoCaptioningModel(ModelAbstract):
    def __init__(self, config: CfgNode) -> None:
        super().__init__(config)

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
        enc_hidden = self.encoder(X)
        output = self.head(enc_hidden, y)
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
        return self.head.beam_search(enc_hidden, max_len, beam_size)

class VideoCaptioningModel_VM(VideoCaptioningModel):
    def __init__(self, config):
        super().__init__(config)
        self.mapper = nn.Linear(576, 768)
    
    def forward(self, X: torch.Tensor , y: Dict[str, torch.Tensor]) -> torch.Tensor:
        enc_hidden = self.encoder(X)
        enc_hidden = self.mapper(enc_hidden)
        output = self.head(enc_hidden, y)
        return output.logits

    def generate(self, X: torch.Tensor, max_len: int = 64, beam_size: int = 1) -> str:
        enc_hidden = self.encoder(X)
        enc_hidden = self.mapper(enc_hidden)
        return self.head.beam_search(enc_hidden, max_len, beam_size)
