from typing import Dict, List

import torch
from torch import nn
from torchmetrics.classification import Accuracy
from fvcore.common.config import CfgNode

from .model_abstract import ModelAbstract
from .encoders import create_encoder, EncoderAbstract
from .heads import create_head, HeadAbstract

class VideoClassificationModel(ModelAbstract):
    def __init__(self, config: CfgNode) -> None:
        super().__init__(config)
        self.acc_calculator = Accuracy(task="multiclass", num_classes=config.MODEL.HEAD.NUM_CLASSES)

    def create_encoder(self) -> EncoderAbstract:
        return create_encoder(self.config)

    def create_head(self) -> HeadAbstract:
        return create_head(self.config)

    def create_loss_function(self) -> nn.Module:
        return nn.CrossEntropyLoss()
            
    def compute_metrics(self, step_outputs: List) -> Dict:
        """
            Compute metrics after each epoch
            Args:
                outputs: list of dictionary contain predictions, labels
        """
        all_probas = torch.cat([out['preds'].detach().cpu() for out in step_outputs])
        all_labels = torch.cat([out['labels'].detach().cpu() for out in step_outputs])
        # print('cat shape:', all_probas.shape, all_labels.shape)
        all_preds = torch.argmax(all_probas, axis=1)
        acc = float(self.acc_calculator(all_preds, all_labels))
        return {'accuracy': acc}
