
import abc
from typing import Any, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import lightning as L
import transformers

class ModelAbstract(abc.ABC, L.LightningModule):
    """Define abstract method for the models"""
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.encoder = self.create_encoder()
        self.downstream_head = self.create_downstream_head()
        self.loss_funct = self.create_loss_function()

    @abc.abstractmethod
    def create_encoder(self) -> nn.Module:
        pass

    @abc.abstractmethod
    def create_downstream_head(self) -> nn.Module:
        pass

    @abc.abstractmethod
    def create_loss_function(self) -> nn.Module:
        pass

    def forward(self, X: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        features = self.encoder(X)
        y_pred = self.downstream_head(features)
        return y_pred

    def compute_loss(self, y_pred, y):
        return self.criterion(y_pred, y)

    def training_step(self, batch: Tuple, *args: Any, **kwargs: Any) -> None:
        X, y = batch
        y_pred = self(X)
        loss = self.compute_loss(y_pred, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        val_loss = self.compute_loss(y_pred, y)
        self.log("val_loss", val_loss)

    @abc.abstractmethod
    def compute_metrics(self, *args: Any, **kwargs: Any):
        return dict()

    def log_metrics(self, metric_dict, phase='train'):
        
        for k, v in metric_dict.items():
            self.log(f'train_{k}', v)

    def training_epoch_end(self, training_step_outputs):
        metric_dict = self.compute_metrics(training_step_outputs)
        self.log_metrics(metric_dict, phase='train')
        
    def validation_epoch_end(self, validation_step_outputs):
        metric_dict = self.compute_metrics(validation_step_outputs)
        self.log_metrics(metric_dict, phase='val')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        optimizer = AdamW(self.backbone.parameters(), lr=self.cfg.init_lr, eps=self.cfg.eps, betas=self.cfg.betas)
        num_train_steps = int(self.cfg.num_train_examples / self.cfg.batch_size * self.cfg.epochs)

        # Defining LR SCheduler
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_train_steps, eta_min=self.cfg.min_lr)

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