import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from fvcore.common.config import CfgNode
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import wandb

from src.models import create_model
from src.models.captioning_model import VideoCaptioningModel
from src.datasets import create_dataset, classification_collate_fn
from src.utils.general import set_deterministic

parser = argparse.ArgumentParser(description="Train a video model")
parser.add_argument("--config", help="The config file", 
                        default="src/config/cap_svt_charades_s224_f8_exp0.yaml")

args = parser.parse_args()


# Load config
config = CfgNode.load_yaml_with_base(args.config)
config = CfgNode(config)

# make reproducible
set_deterministic(config.SEED)
lit_module = create_model(config)

tokenizer = lit_module.head.tokenizer

caption = ["The man is eating a cake", "I dont have time"]
tokens_and_masks = tokenizer(caption, return_tensors="pt", padding=True)
print(tokens_and_masks['input_ids'].shape, tokens_and_masks['attention_mask'].shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lit_module.to(device)


X = torch.rand(len(caption), 8, 3, 224, 224).to(device)
y = {
     'input_ids': tokens_and_masks['input_ids'].to(device),
     'attention_mask': tokens_and_masks['attention_mask'].to(device)
     }

pred = lit_module(X, y)
print(pred.shape) # bs, seq len, vocab size

# compute loss
batch = X,y
loss = lit_module.training_step(batch)
print('Loss:', loss)

generated_cap = lit_module.generate(X[[0]], max_len=64, beam_size=1)

print("Generated cap:", generated_cap)

