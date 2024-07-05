import torch
from torch import nn
from fvcore.common.config import CfgNode

from .head_abstract import HeadAbstract

class Dense(nn.Module):
    def __init__(self, inp_size: int, out_size: int, dropout: float = 0.0, layer_norm: bool = False):
        super().__init__()
        layers = [nn.Linear(inp_size, out_size), nn.ReLU()]
        if layer_norm:
            layers.append(nn.LayerNorm(out_size))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.layer = nn.Sequential(*layers)
        
    def forward(self, X: torch.Tensor):
        return self.layer(X)

class MLPHead(HeadAbstract):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.config = config
        
        layers = []
        in_size = self.config.MODEL.ENCODER.HIDDEN_SIZE
        dropout = self.config.MODEL.HEAD.DROPOUT if hasattr(self.config.MODEL.HEAD, 'DROPOUT') else 0.0
        batch_norm = self.config.MODEL.HEAD.BATCH_NORM if hasattr(self.config.MODEL.HEAD, 'BATCH_NORM') else False
        
        for out_size in self.config.MODEL.HEAD.LAYERS[1:]:
            layers.append(Dense(in_size, out_size, dropout, batch_norm))
            in_size = out_size

        self.mlp = nn.Sequential(*layers)
        
        out_size = self.config.MODEL.HEAD.NUM_CLASSES
        self.classifier = nn.Linear(in_size, out_size)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None):
        out = self.mlp(X)
        out = self.classifier(out)
        return out