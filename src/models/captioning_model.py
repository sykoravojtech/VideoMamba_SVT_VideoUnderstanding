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

class VideoCaptioningModel(ModelAbstract):
    def __init__(self, config: CfgNode) -> None:
        super().__init__(config)
        self.visual_mapper = nn.Linear(3072, config.MODEL.ENCODER.HIDDEN_SIZE)

    def create_encoder(self) -> EncoderAbstract:
        encoder = create_encoder(self.config)
        for param in encoder.parameters():
            param.requires_grad = False
        return encoder

    def create_head(self) -> HeadAbstract:
        head = create_head(self.config)
        if(self.config.MODEL.HEAD.FROZEN):
            for param in head.parameters():
                param.requires_grad = False
        return head

    def forward(self, X: torch.Tensor , y: Dict[str, torch.Tensor]) -> torch.Tensor:
        '''
            Args:
                X: video pixel tensor
                y: dictionary of {'input_ids', 'attention_mask'} for the target caption
            Returns:
                logits: the output logits of the model
        '''
        ## Encodings preloaded
        #enc_hidden = self.encoder(X)

        if(self.config.MODEL.LINEAR_MAP):
            # Apply the visual mapper
            X = self.visual_mapper(X)

        output = self.head(X, y)
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
        if(self.config.MODEL.LINEAR_MAP):
            enc_hidden = self.visual_mapper(enc_hidden)
        return self.head.beam_search(enc_hidden, max_len, beam_size)
        
    def configure_optimizers(self):                        
        if(self.config.MODEL.LINEAR_MAP):
            if(self.config.MODEL.HEAD.FROZEN):
                optimizer = AdamW(self.visual_mapper.parameters(), 
                                lr=self.config.TRAIN.OPTIM.INIT_LEARNING_RATE, 
                                eps=self.config.TRAIN.OPTIM.EPS, betas=self.config.TRAIN.OPTIM.BETAS)
            else:
                optimizer = AdamW(list(self.visual_mapper.parameters()) + list(self.head.parameters()), 
                                lr=self.config.TRAIN.OPTIM.INIT_LEARNING_RATE, 
                                eps=self.config.TRAIN.OPTIM.EPS, betas=self.config.TRAIN.OPTIM.BETAS)
        else:
            ptimizer = AdamW(self.head.parameters(), lr=self.config.TRAIN.OPTIM.INIT_LEARNING_RATE, 
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