import os

import torch
from torch.utils.data import DataLoader
from fvcore.common.config import CfgNode
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import wandb

from src.models import create_model
from src.datasets import create_dataset, classification_collate_fn

CONFIG_FILE = 'src/config/cls_svt_s224_f8.yaml'

def train():
    # config = CfgNode()
    config = CfgNode.load_yaml_with_base(CONFIG_FILE)
    config = CfgNode(config)

    wandb.login(key=config.WANDB_KEY)
    wandb.init(project=config.PROJECT_NAME,
                name=config.EXPERIMENT,
                group=config.MODEL.TYPE)

    dataset = create_dataset(config)
    train_dataset = dataset.get_train_dataset()
    val_dataset = dataset.get_val_dataset()

    batch_size = config.TRAIN.BATCH_SIZE
    num_workers = config.DATA.NUM_WORKERS
    train_loader = DataLoader(train_dataset,batch_size=batch_size,
                                               pin_memory=True,drop_last=True,num_workers=num_workers,
                                               collate_fn=classification_collate_fn)
    valid_loader = DataLoader(val_dataset,batch_size=batch_size,num_workers=num_workers,
                                               pin_memory=True,drop_last=False,
                                               collate_fn=classification_collate_fn)
    lit_module = create_model(config)
    
    output_dir = os.path.join(config.OUTPUT_DIR, config.EXPERIMENT)
    wandb_logger = WandbLogger(project=config.PROJECT_NAME)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpointer = ModelCheckpoint(
         dirpath=os.path.join(output_dir,),
         filename='Ep{epoch}-{val_loss:.3f}-{val_accuracy:.3f}',
         monitor='val_loss',
         verbose=True,
         save_weights_only=True
    )
    
    trainer = Trainer(default_root_dir=output_dir, precision=16, max_epochs=config.TRAIN.NUM_EPOCHS,
                     check_val_every_n_epoch=1, enable_checkpointing=True,
                     log_every_n_steps=config.TRAIN.LOG_STEPS,
                     logger=wandb_logger,
                     callbacks=[lr_monitor, checkpointer],
                     accelerator=config.TRAIN.ACCELERATOR, devices=config.TRAIN.DEVICES)
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=valid_loader)
   

if __name__ == '__main__':
    train()
