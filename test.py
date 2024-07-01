import os
import argparse

import torch
from torch.utils.data import DataLoader
from fvcore.common.config import CfgNode
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import wandb

from src.models import create_model
from src.datasets import create_dataset, classification_collate_fn, captioning_collate_fn
from src.utils.general import set_deterministic

WEIGHT = "runs/cls_svt_charades_s224_f8_exp0/epoch=6-val_mAP=0.093.ckpt"

parser = argparse.ArgumentParser(description="Test a video model")
parser.add_argument("--config", help="The config file", 
                        default="src/config/cls_svt_ucf101_s224_f8_exp0.yaml")

args = parser.parse_args()

def get_collate_fn(config: CfgNode):
    if config.MODEL.TYPE == 'classification':
        return classification_collate_fn(config)
    elif config.MODEL.TYPE == 'captioning':
        return captioning_collate_fn(config)
    else:
        raise ValueError("Invalid model type")

def test():
    """Train a new model"""

    # Load config
    config = CfgNode.load_yaml_with_base(args.config)
    config = CfgNode(config)

    # make reproducible
    set_deterministic(config.SEED)

    # create dataset
    dataset = create_dataset(config)
    train_dataset = dataset.get_train_dataset()
    val_dataset = dataset.get_val_dataset()

    # create dataloaders
    batch_size = config.TRAIN.BATCH_SIZE
    num_workers = config.DATA.NUM_WORKERS
    collate_fn = get_collate_fn(config)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,
                                              pin_memory=True,drop_last=True,num_workers=num_workers,
                                               collate_fn=collate_fn)
    valid_loader = DataLoader(val_dataset,batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True,drop_last=False,
                                collate_fn=collate_fn)
    # crete model
    lit_module = create_model(config, WEIGHT)

    # callbacks
    output_dir = os.path.join(config.OUTPUT_DIR, config.EXPERIMENT)
    
    # trainer
    trainer = Trainer(default_root_dir=output_dir,
                     accelerator=config.TRAIN.ACCELERATOR, devices=config.TRAIN.DEVICES)

    # training
    results = trainer.validate(model=lit_module, dataloaders=valid_loader)
    # results = trainer.predict(lit_module, valid_loader)

    print(results)

    # preds = []
    # gts = []
    # for p, g in results:
    #     preds.append(p)
    #     gts.append(g)
    # preds  = torch.cat(preds)
    # gts = torch.cat(gts)
    # torch.save( preds, "preds.pth")
    # torch.save(gts, "gts.pth")

if __name__ == '__main__':
    test()
