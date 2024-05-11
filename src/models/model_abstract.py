
import abc
from typing import Any, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from lightning import LightningModule

from ..utils.general import free_subnet

class ModelAbstract(abc.ABC, LightningModule):
    """Define abstract method for the models"""
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.encoder = self.create_encoder()
        if self.config.TRAIN.FREEZE_ENCODER:
            free_subnet(self.encoder)
        self.head = self.create_head()
        self.loss_func = self.create_loss_function()
        self.training_step_outputs = []
        self.validation_step_outputs = []

    @abc.abstractmethod
    def create_encoder(self) -> nn.Module:
        pass

    @abc.abstractmethod
    def create_head(self) -> nn.Module:
        pass

    @abc.abstractmethod
    def create_loss_function(self) -> nn.Module:
        pass

    def forward(self, X: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        features = self.encoder(X)
        print("feature shape:", features.shape)
        y_pred = self.head(features)
        print('head out shape:', y_pred.shape)
        return y_pred

    def compute_loss(self, y_pred, y):
        return self.loss_func(y_pred, y)

    def training_step(self, batch: Tuple, batch_idx) -> torch.Tensor:
        X, y = batch
        y_pred = self(X)
        # print('y shape:', y.shape)
        # print('y_pred:', y_pred.shape)
        loss = self.compute_loss(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.training_step_outputs.append({'y_pred': y_pred, 'y_true':y})
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        val_loss = self.compute_loss(y_pred, y)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.append(y_pred)

    @abc.abstractmethod
    def compute_metrics(self, *args: Any, **kwargs: Any):
       return dict()

    def log_metrics(self, metric_dict, phase='train'):
        for k, v in metric_dict.items():
            self.log(f'{phase}_{k}', v)

    def on_train_epoch_end(self):
        metric_dict = self.compute_metrics(self.training_step_outputs)
        self.log_metrics(metric_dict, phase='train')
        self.training_step_outputs.clear()  # free memory
        
    def on_validation_epoch_end(self):
        metric_dict = self.compute_metrics(self.validation_step_outputs)
        self.log_metrics(metric_dict, phase='val')
        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = AdamW(self.head.parameters(), lr=self.config.TRAIN.OPTIM.INIT_LEARNING_RATE, 
                        eps=self.config.TRAIN.OPTIM.EPS, betas=self.config.TRAIN.OPTIM.BETAS)

        # Defining LR SCheduler
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.config.TRAIN.OPTIM.MAX_LR_STEPS,
                                        eta_min=self.config.TRAIN.OPTIM.MIN_LEARNING_RATE)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                # "monitor": "metric_to_track",
                # "frequency": "indicates how often the metric is updated"
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }