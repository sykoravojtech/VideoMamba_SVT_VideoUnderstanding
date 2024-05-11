from torch import nn
from fvcore.common.config import CfgNode

from .head_abstract import HeadAbstract

class GenerativeHead(HeadAbstract):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.config = config

    def forward(self, X):
        return None