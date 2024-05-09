
from .downstream_head_abstract import DownstreamHeadAbstract

class FullyConnectedHead(DownstreamHeadAbstract):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def forward(self, X):
        return None