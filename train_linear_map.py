import os
import argparse
from glob import glob
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from fvcore.common.config import CfgNode
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import wandb

from src.datasets import create_dataset, classification_collate_fn, captioning_collate_fn
from src.utils.general import set_deterministic
from src.models.captioning_model_linear_map import VideoCaptioningModelLinearMap
from src.models.encoders import EncoderAbstract



parser = argparse.ArgumentParser(description="Train a video model")
parser.add_argument("-c", "--config", help="The config file", 
                        default="src/config/cap_lm_svt_charades_llama_exp0.yaml")

args = parser.parse_args()

class VideoCaptioningModelHead(VideoCaptioningModelLinearMap):
    def __init__(self, config: CfgNode) -> None:
        super().__init__(config)

def get_collate_fn(config: CfgNode):
    if config.MODEL.TYPE == 'classification':
        return classification_collate_fn(config)
    elif config.MODEL.TYPE == 'captioning':
        return captioning_collate_fn(config)
    else:
        raise ValueError("Invalid model type")

class CaptioningDataset(Dataset):
    def __init__(self, train: bool):
        super().__init__()
        if train:
            self.x = sorted(glob("data/encodings_2/train_x_*.pt"))
            self.y = sorted(glob("data/encodings_2/train_y_*.pt"))
        else:
            self.x = sorted(glob("data/encodings_2/val_x_*.pt"))
            self.y = sorted(glob("data/encodings_2/val_y_*.pt"))
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, ind):
        return torch.load(self.x[ind]).flatten(), torch.load(self.y[ind])

def train():
    """Train a new model"""

    # Load config
    config = CfgNode.load_yaml_with_base(args.config)
    config = CfgNode(config)

    # make reproducible
    set_deterministic(config.SEED)

    # Wandb init
    wandb.login(key=config.WANDB_KEY)
    wandb.init(project=config.PROJECT_NAME,            name=config.EXPERIMENT,         group=config.MODEL.TYPE)

    # create dataset
    # dataset = create_dataset(config)
    train_dataset = CaptioningDataset(True) # dataset.get_train_dataset()
    val_dataset = CaptioningDataset(False) # dataset.get_val_dataset()

    # create dataloaders
    batch_size = config.TRAIN.BATCH_SIZE
    num_workers = config.DATA.NUM_WORKERS
    # collate_fn = get_collate_fn(config)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,
                                              pin_memory=True,drop_last=True,num_workers=num_workers,
                                               )
    valid_loader = DataLoader(val_dataset,batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True,drop_last=False,
                                )
    # create model
    lit_module = VideoCaptioningModelHead(config)

    # callbacks
    output_dir = os.path.join(config.OUTPUT_DIR, config.EXPERIMENT)
    wandb_logger = WandbLogger(project=config.PROJECT_NAME)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_monitor_criterion = config.TRAIN.BEST_CHECKPOINT_BY
    checkpointer = ModelCheckpoint(
         dirpath=os.path.join(output_dir,),
         filename='{epoch:}-{%s:.3f}'%(checkpoint_monitor_criterion),
         monitor=checkpoint_monitor_criterion,
         verbose=True,
         save_weights_only=True,
         mode='min' if 'loss' in checkpoint_monitor_criterion else 'max',
         save_last=True
    )
    
    # trainer
    trainer = Trainer(default_root_dir=output_dir, precision=config.TRAIN.PRECISION, max_epochs=config.TRAIN.NUM_EPOCHS,
                     check_val_every_n_epoch=1, enable_checkpointing=True,
                     log_every_n_steps=config.TRAIN.LOG_STEPS,
                     logger=wandb_logger,
                     callbacks=[lr_monitor, checkpointer],
                     accelerator=config.TRAIN.ACCELERATOR, devices=config.TRAIN.DEVICES)

    # training
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=valid_loader)
   

if __name__ == '__main__':
    train()
