from typing import Dict
import os
import argparse
from glob import glob
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from fvcore.common.config import CfgNode
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
# from lightning.pytorch.loggers import WandbLogger
# import wandb

from src.models.captioning_model import VideoCaptioningModel
# from src.datasets import create_dataset
from src.utils.general import set_deterministic

parser = argparse.ArgumentParser(description="Train a video model")
parser.add_argument("-c", "--config", help="The config file", 
                        default="src/config/cls_svt_ucf101_s224_f8_exp0.yaml")

args = parser.parse_args()

class CaptioningDataset(Dataset):
    def __init__(self, train: bool, mean: torch.tensor, std: torch.tensor, frame_skip: int = 8):
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.frame_skip = frame_skip
        if train:
            self.x = sorted(glob("data/encodings/inp_64_int/train_x_*.pt"))
            self.y = sorted(glob("data/encodings/inp_64_int/train_y_*.pt"))
        else:
            self.x = sorted(glob("data/encodings/inp_64_int/val_x_*.pt"))
            self.y = sorted(glob("data/encodings/inp_64_int/val_y_*.pt"))
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, ind):
        return ((torch.load(self.x[ind])[::self.frame_skip].permute(0,2,3,1)/255 - self.mean)/self.std).permute(0,3,1,2), torch.load(self.y[ind])


class VideoCaptioningModel_(VideoCaptioningModel):
    def __init__(self, config):
        super().__init__(config)
        self.mapper = nn.Linear(576, 768)
    
    def forward(self, X: torch.Tensor , y: Dict[str, torch.Tensor]) -> torch.Tensor:
        '''
            Args:
                X: video pixel tensor
                y: dictionary of {'input_ids', 'attention_mask'} for the target caption
            Returns:
                logits: the output logits of the model
        '''
        enc_hidden = self.encoder(X)
        enc_hidden = self.mapper(enc_hidden)
        output = self.head(enc_hidden, y)
        return output.logits

    def generate(self, X: torch.Tensor, max_len: int = 64, beam_size: int = 1) -> str:
        enc_hidden = self.encoder(X)
        enc_hidden = self.mapper(enc_hidden)
        return self.head.beam_search(enc_hidden, max_len, beam_size)

def train():
    """Train a new model"""

    # Load config
    config = CfgNode.load_yaml_with_base(args.config)
    config = CfgNode(config)

    # make reproducible
    set_deterministic(config.SEED)

    # Wandb init
    # wandb.login(key=config.WANDB_KEY)
    # wandb.init(project=config.PROJECT_NAME,
    #             name=config.EXPERIMENT,
    #             group=config.MODEL.TYPE)

    # create dataset
    train_dataset = CaptioningDataset(True, config.DATA.MEAN, config.DATA.STD, config.DATA.FRAME_SKIP)
    val_dataset = CaptioningDataset(False, config.DATA.MEAN, config.DATA.STD, config.DATA.FRAME_SKIP)

    # create dataloaders
    batch_size = config.TRAIN.BATCH_SIZE
    num_workers = config.DATA.NUM_WORKERS
    train_loader = DataLoader(train_dataset,batch_size=batch_size,
                                              pin_memory=True,drop_last=True,num_workers=num_workers,)
    valid_loader = DataLoader(val_dataset,batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True,drop_last=False,)
    # crete model
    lit_module = VideoCaptioningModel_(config)
    # lit_module.encoder.vit.time_embed = torch.nn.Parameter(torch.concat([lit_module.encoder.vit.time_embed.data]*4, axis=1))

    # callbacks
    output_dir = os.path.join(config.OUTPUT_DIR, config.EXPERIMENT)
    # wandb_logger = WandbLogger(project=config.PROJECT_NAME)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # checkpoint_monitor_criterion = config.TRAIN.BEST_CHECKPOINT_BY
    checkpointer = ModelCheckpoint(
         dirpath=os.path.join(output_dir,),
        #  filename='{epoch:}-{%s:.3f}'%(checkpoint_monitor_criterion),
        #  monitor=checkpoint_monitor_criterion,
         verbose=True,
         save_weights_only=True,
        #  mode='min' if 'loss' in checkpoint_monitor_criterion else 'max',
         save_last=True,
         every_n_epochs=1,
         save_top_k=-1
    )
    
    # trainer
    trainer = Trainer(default_root_dir=output_dir, 
                    # precision=config.TRAIN.PRECISION, 
                    max_epochs=config.TRAIN.NUM_EPOCHS,
                     check_val_every_n_epoch=1, enable_checkpointing=True,
                    #  log_every_n_steps=config.TRAIN.LOG_STEPS,
                    #  logger=wandb_logger,
                     callbacks=[lr_monitor, checkpointer],
                     accelerator=config.TRAIN.ACCELERATOR, devices=config.TRAIN.DEVICES)

    # training
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=valid_loader)
   

if __name__ == '__main__':
    train()
