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

class HiddenWiseReducer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, X: torch.Tensor):
        out = self.linear(X)
        return out[:,:,0]

class TokenWiseReducer(nn.Module):
    def __init__(self, num_tokens: int, learnable_weights: bool = True):
        super().__init__()
        self.num_tokens = num_tokens
        self.learnable_weights = learnable_weights
        if learnable_weights:
            self.weight = nn.Parameter(torch.ones(1, num_tokens, 1))
        
    def forward(self, X: torch.Tensor):
        if self.learnable_weights:
            out = (self.weight * X).sum(axis=1) / self.num_tokens
        else:
            out = X.mean(axis=1)
        return out

class MLPHead(HeadAbstract):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.config = config
        
        layers = []
        hidden_size = self.config.MODEL.ENCODER.HIDDEN_SIZE
        dropout = self.config.MODEL.HEAD.DROPOUT if hasattr(self.config.MODEL.HEAD, 'DROPOUT') else 0.0
        layer_norm = self.config.MODEL.HEAD.LAYER_NORM if hasattr(self.config.MODEL.HEAD, 'LAYER_NORM') else False
        self.return_all_hiddens = self.config.MODEL.ENCODER.RETURN_ALL_HIDDEN
        num_visual_tokens = self.config.MODEL.ENCODER.NUM_VISUAL_TOKENS
        self.reducer_type = self.config.MODEL.HEAD.REDUCER_TYPE
        in_size = num_visual_tokens

        if self.return_all_hiddens:
            if self.reducer_type == 'weighted_hidden_wise':                
                self.reducer = HiddenWiseReducer(hidden_size)
                in_size = num_visual_tokens
            elif self.reducer_type == 'weighted_token_wise':
                self.reducer = TokenWiseReducer(num_visual_tokens, learnable_weights=True)
                in_size = hidden_size
            elif self.reducer_type == 'average':
                self.reducer = TokenWiseReducer(num_visual_tokens, learnable_weights=False)
                in_size = hidden_size
            else:
                raise ValueError(f"Unknown reducer type: {self.reducer_type}")
        else:
            self.reducer = nn.Identity()
            in_size = hidden_size

        for out_size in self.config.MODEL.HEAD.LAYERS[1:]:
            layers.append(Dense(in_size, out_size, dropout, layer_norm))
            in_size = out_size

        self.mlp = nn.Sequential(*layers)
        
        out_size = self.config.MODEL.HEAD.NUM_CLASSES
        self.classifier = nn.Linear(in_size, out_size)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None):
        out = self.reducer(X)
        
        out = self.mlp(out)
        out = self.classifier(out)
        return out