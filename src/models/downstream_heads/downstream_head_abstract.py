import abc

from torch import nn

class DownstreamHeadAbstract(abc.ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, X):
        pass