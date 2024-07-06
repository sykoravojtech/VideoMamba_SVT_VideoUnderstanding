import os
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from fvcore.common.config import CfgNode
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorchvideo.data.encoded_video import EncodedVideo

from src.utils.visualizations import investigate_video, display_gif
from src.datasets.transformations import get_train_transforms, get_val_transforms

from src.models import create_model
from src.utils.general import set_deterministic

parser = argparse.ArgumentParser(description="Train a video model")
# parser.add_argument("--config", help="The config file", 
#                         default="src/config/cls_svt_charades_s224_f8_exp0.yaml")
parser.add_argument("--config", help="The config file", 
                        default="src/config/cls_vm_charades_s224_f8_exp0.yaml")

args = parser.parse_args()

# Load config
config = CfgNode.load_yaml_with_base(args.config)
config = CfgNode(config)

DATA_DIR = "data/raw/Charades"
VIDEO_DIRS = f"{DATA_DIR}/videos"
CSV_PATH = f"{DATA_DIR}/Charades_v1_test.csv"
# HEAD_WEIGHT = "runs/cls_svt_charades_s224_f8_exp0/epoch=14-val_mAP=0.158.ckpt"
HEAD_WEIGHT = "runs/cls_vm_charades_s224_f8_exp0/epoch=38-val_mAP=0.204.ckpt"


action_map = pd.read_csv(f"{DATA_DIR}/Charades_v1_classes_new_map.csv")
action_id2text = action_map.set_index('action_id')['action'].to_dict()
action_label2text = action_map.set_index('label')['action'].to_dict()

df = pd.read_csv(CSV_PATH)
# sample =  df.iloc[0]
sample = df.sample(1).iloc[0]

lit_module = create_model(config)
# lit_module = VideoCaptioningModel.load_from_checkpoint(WEIGHT)

head_state_dict = torch.load(HEAD_WEIGHT, map_location='cpu')['state_dict']

load_info = lit_module.load_state_dict(head_state_dict, strict=False)
# print(load_info)

lit_module.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lit_module.to(device)

# Load video
VID_PATH = f"data/raw/Charades/videos/{sample['id']}.mp4"
video = EncodedVideo.from_path(VID_PATH)

actions = sample.actions.split(";")

# Compose video data transforms
transform = get_val_transforms(config)

# Print ground truth actions
for gt_action in actions:
    current_action, start_time, end_time = gt_action.split(' ')
    current_action = action_id2text[current_action]
    print(f"Ground truth action: {current_action}. Start time: {start_time}. End time: {end_time}")

video_duration = float(video._duration)

all_clip_tensors = []

for clip_start_sec in np.arange(0, video_duration, config.DATA.CLIP_DURATION):
    clip_duration = config.DATA.CLIP_DURATION
    clip_end_sec = min(clip_start_sec + clip_duration, video_duration)

    clip_data = video.get_clip(start_sec=clip_start_sec, end_sec=clip_start_sec + clip_duration)
    clip_data = transform(clip_data)
    clip_tensor = clip_data['video']  # (C, T, H, W)

    all_clip_tensors.append(clip_tensor)

# concat clip tensors at the time dimension to write to a gif
whole_video_tensor = torch.cat(all_clip_tensors, dim=1)
gif_save_path = "assets/charades_test_model.gif"
display_gif(whole_video_tensor, gif_save_path, config.DATA.MEAN, config.DATA.STD, gif_duration=10)

# Prepare video tensor for inference
inp_video_tensors = torch.stack(all_clip_tensors, dim=0)  # (B, C, T, H, W)
inp_video_tensors = inp_video_tensors.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) -> (B, T, C, H, W)

print()
with torch.no_grad():
    inp_video_tensors = inp_video_tensors.to(device)
    model_output = lit_module(inp_video_tensors).sigmoid().cpu().numpy()
    model_output = model_output.max(axis=0)
    topk_class_indices = np.argsort(model_output)[-5:][::-1]
    for ind in topk_class_indices:
        print(f"Predicted action: {action_label2text[ind]}. Probability: {model_output[ind]:.2f}")