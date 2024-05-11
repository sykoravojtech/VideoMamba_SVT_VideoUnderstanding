import numpy as np
import torch
from torch import nn
from torchmetrics.classification import Accuracy

from .model_abstract import ModelAbstract
from .encoders import create_encoder, EncoderAbstract
from .heads import create_head, HeadAbstract

class VideoClassificationModel(ModelAbstract):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.acc_calculator = Accuracy(task="multiclass", num_classes=config.MODEL.HEAD.NUM_CLASSES)

    def create_encoder(self) -> EncoderAbstract:
        return create_encoder(self.config)

    def create_head(self) -> HeadAbstract:
        return create_head(self.config)

    def create_loss_function(self) -> nn.Module:
        return nn.CrossEntropyLoss()
            
    def compute_metrics(self, outputs):
        all_probas = np.concatenate([out['preds'].detach().cpu().numpy() for out in outputs])
        all_labels = np.concatenate([out['labels'].detach().cpu().numpy() for out in outputs])
        all_preds = np.argmax(all_probas, axis=1)
        acc = float(self.acc_calculator(y_true=all_labels, y_pred=all_preds))
        return {'accuracy': acc}
