import abc

import torch
from torch import nn

class EncoderAbstract(abc.ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, X: torch.Tensor, y: torch.Tensor):
        return X

