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
                        default="src/config/cls_svt_charades_s224_f8_exp0.yaml")

args = parser.parse_args()

BATCH_SIZE = 16 # How many clip's encoding should be saved in a pth file

def get_collate_fn(config: CfgNode):
    if config.MODEL.TYPE == 'classification':
        return classification_collate_fn(config)
    elif config.MODEL.TYPE == 'captioning':
        return captioning_collate_fn(config)
    else:
        raise ValueError("Invalid model type")

def create_encodings():
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
    num_workers = config.DATA.NUM_WORKERS
    collate_fn = get_collate_fn(config)
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,
                                shuffle=False,pin_memory=True,num_workers=num_workers,collate_fn=collate_fn,prefetch_factor=4)
    valid_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,
                                num_workers=num_workers,
                                pin_memory=True,
                                shuffle=False,
                                collate_fn=collate_fn,prefetch_factor=4)
    # crete model
    lit_module = create_model(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lit_module = lit_module.to(device).eval()

    # callbacks
    output_dir = config.DATA.ENCODING_DIR
    if os.path.exists(output_dir):
        raise IOError('Warning: {output_dir} already exists. Please remove it if you want to continue')
    os.makedirs(output_dir, exist_ok=True)
    print(f'Creating the video encoded features at {output_dir}...')
    
    with torch.no_grad():
        for i, (X, y) in tqdm(enumerate(train_loader)):
            X = X.to(device)
            enc = lit_module.encoder(X)
            torch.save(enc.cpu().detach(), os.path.join(output_dir, f"train_x_{i}.pt"))
            torch.save(y, os.path.join(output_dir, f"train_y_{i}.pt"))

    with torch.no_grad():
        for i, (X, y) in tqdm(enumerate(valid_loader)):
            X = X.to(device)
            enc = lit_module.encoder(X)
            torch.save(enc.cpu().detach(), os.path.join(output_dir, f"val_x_{i}.pt"))
            torch.save(y, os.path.join(output_dir, f"val_y_{i}.pt"))
   

if __name__ == '__main__':
    create_encodings()