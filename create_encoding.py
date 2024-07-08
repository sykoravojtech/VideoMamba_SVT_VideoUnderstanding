import os
import argparse

import torch
from torch.utils.data import DataLoader
from fvcore.common.config import CfgNode
from tqdm import tqdm
from src.models import create_model
from src.datasets import create_dataset, classification_collate_fn, captioning_collate_fn
from src.utils.general import set_deterministic

parser = argparse.ArgumentParser(description="Train a video model")
parser.add_argument("-c", "--config", help="The config file", 
                        default="src/config/cls_svt_ucf101_s224_f8_exp0.yaml")

args = parser.parse_args()

def get_collate_fn(config: CfgNode):
    if config.MODEL.TYPE == 'classification':
        return classification_collate_fn(config)
    elif config.MODEL.TYPE == 'captioning':
        return captioning_collate_fn(config)
    else:
        raise ValueError("Invalid model type")

def train():
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
    batch_size = 1
    num_workers = config.DATA.NUM_WORKERS
    collate_fn = get_collate_fn(config)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,
                                shuffle=False,pin_memory=True,num_workers=num_workers,collate_fn=collate_fn,prefetch_factor=4)
    valid_loader = DataLoader(val_dataset,batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True,
                                shuffle=False,
                                collate_fn=collate_fn,prefetch_factor=4)
    # crete model
    lit_module = create_model(config)
    lit_module = lit_module.to("cuda").eval()

    # callbacks
    output_dir = os.path.join("data/encodings_2/")
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, (X, y) in tqdm(enumerate(train_loader)):
            X = X.reshape((batch_size * config.DATA.NUM_SAMPLED_FRAMES_MULT, 8, *X.shape[2:]))
            X = X.to("cuda")
            enc = lit_module.encoder(X)
            torch.save(enc.cpu().detach().squeeze(), os.path.join(output_dir, f"train_x_{i}.pt"))
            torch.save(y, os.path.join(output_dir, f"train_y_{i}.pt"))

    with torch.no_grad():
        for i, (X, y) in tqdm(enumerate(valid_loader)):
            X = X.reshape((batch_size * config.DATA.NUM_SAMPLED_FRAMES_MULT, 8, *X.shape[2:]))
            X = X.to("cuda")
            enc = lit_module.encoder(X)
            torch.save(enc.cpu().detach().squeeze(), os.path.join(output_dir, f"val_x_{i}.pt"))
            torch.save(y, os.path.join(output_dir, f"val_y_{i}.pt"))
   

if __name__ == '__main__':
    train()
