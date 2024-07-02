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

from src.utils.visualizations import investigate_video, display_gif
from src.datasets.transformations import get_train_transforms, get_val_transforms


from src.models import create_model
from src.models.captioning_model import VideoCaptioningModel
from src.datasets import create_dataset, classification_collate_fn, captioning_collate_fn
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
# lit_module = create_model(config)

import glob
WEIGHT = glob.glob('runs/cap_svt_charades_s224_f8_exp0/epoch=*.ckpt')[0]
lit_module = VideoCaptioningModel.load_from_checkpoint(WEIGHT)

tokenizer = lit_module.head.tokenizer

caption = ["The man is eating a cake", "I dont have time"]
tokens_and_masks = tokenizer(caption, return_tensors="pt", padding=True)
print(tokens_and_masks['input_ids'].shape, tokens_and_masks['attention_mask'].shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lit_module.to(device)
lit_module.eval()

dataset = create_dataset(config)
train_dataset = dataset.get_train_dataset()
val_dataset = dataset.get_val_dataset()

batch_size = 1
num_workers = config.DATA.NUM_WORKERS
collate_fn = captioning_collate_fn
train_loader = DataLoader(train_dataset,batch_size=batch_size, 
                                             pin_memory=True,drop_last=True,num_workers=num_workers,
                                             collate_fn=collate_fn)
val_loader = DataLoader(val_dataset,batch_size=batch_size,num_workers=num_workers, 
                                             pin_memory=True,drop_last=False,
                                             collate_fn=collate_fn)

X, y = next(iter(val_loader))
X = X.to(device)
for k,v in y.items():
     y[k] = v.to(device)

pred = lit_module(X, y)
print(pred.shape) # bs, seq len, vocab size

# compute loss
batch = X,y
loss = lit_module.training_step(batch)
print('Loss:', loss)

generated_cap = lit_module.generate(X, max_len=128, beam_size=3)

print(len(generated_cap))

print("Generated cap:", generated_cap)

print('True cap:', tokenizer.decode(y['input_ids'][0].cpu(), skip_special_tokens=True))


sample_video = next(iter(val_dataset))
print(sample_video.keys())
# print(sample_video['label'])
# investigate_video(sample_video, ds.get_id2label())
# print(len(ds.get_id2label()))

video_tensor = sample_video["video"]
save_path =  "assets/sample_charades_test.gif"
print("Save sample to:", save_path)


display_gif(video_tensor, save_path, config.DATA.MEAN, config.DATA.STD)

