import torch
from torch import nn
from fvcore.common.config import CfgNode

from .head_abstract import HeadAbstract

class Dense(nn.Module):
    def __init__(self, inp_size: int, out_size: int):
        super().__init__()
        self.layer = nn.Sequential(
                            nn.Linear(inp_size, out_size),
                            nn.ReLU(),
                            nn.LayerNorm(out_size))
    def forward(self, X: torch.Tensor):
        return self.layer(X)


class MLPHead(HeadAbstract):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.config = config
        
        layers = []
        in_size = self.config.MODEL.ENCODER.HIDDEN_SIZE
        for out_size in self.config.MODEL.HEAD.LAYERS[1:]:
            layers.append(Dense(in_size, out_size))
            in_size = out_size

        self.mlp = nn.Sequential(*layers)
        
        out_size = self.config.MODEL.HEAD.NUM_CLASSES
        self.classifier = nn.Linear(in_size, out_size)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None):
        out = self.mlp(X)
        out = self.classifier(out)
        return out