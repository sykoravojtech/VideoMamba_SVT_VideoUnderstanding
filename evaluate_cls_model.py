import os
import argparse

import multiprocessing
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fvcore.common.config import CfgNode
from pytorchvideo.data.encoded_video import EncodedVideo

from src.utils.visualizations import investigate_video, display_gif
from src.datasets.transformations import get_val_transforms

from src.models import create_model
from src.utils.metrics import compute_multilabel_mAP

parser = argparse.ArgumentParser(description="Train a video model")
# parser.add_argument("--config", help="The config file", 
#                         default="src/config/cls_svt_charades_s224_f8_exp0.yaml")
parser.add_argument("--config", help="The config file", 
                        default="src/config/cls_svt_charades_s224_f8_exp0.yaml")
parser.add_argument("--weight", help="The path to the trained weight .ckpt file", 
                        default="checkpoints/cls_svt_charades_s224_f8_exp0/epoch=18-val_mAP=0.165.ckpt")

args = parser.parse_args()

# Load config
config = CfgNode.load_yaml_with_base(args.config)
config = CfgNode(config)

DATA_DIR = config.DATA.ROOT_PATH
VIDEO_DIRS = f"{DATA_DIR}/videos"
CSV_PATH = f"{DATA_DIR}/Charades_v1_test.csv"
ASSET_DIR = "assets"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRIDE = 3 # None means to use the stride = clip duration, i.e. no overlap. Else, set to an integer value.

NUM_PROCESSES = min(multiprocessing.cpu_count(), config.DATA.NUM_WORKERS)  # Define the number of tasks you want to run in parallel
print('Number of CPUs:', NUM_PROCESSES)

action_map = pd.read_csv(f"{DATA_DIR}/Charades_v1_classes_new_map.csv")
action_id2text = action_map.set_index('action_id')['action'].to_dict()
action_label2text = action_map.set_index('label')['action'].to_dict()
action_id2label = action_map.set_index('action_id')['label'].to_dict()

df = pd.read_csv(CSV_PATH)
# df = df.sample(100, random_state=42)

def load_model(config, weight_path):
    if config.DATA.ENCODING_DIR: # head-only weights
        lit_module = create_model(config)
        head_state_dict = torch.load(weight_path, map_location='cpu')['state_dict']
        load_info = lit_module.load_state_dict(head_state_dict, strict=False)
    else: # load full model weights
        lit_module = create_model(config, weight_path=weight_path)

    lit_module.eval()
    
    lit_module.to(DEVICE)
    return lit_module

# load model
lit_module = load_model(config, args.weight)

# some constant object for inference
transform = get_val_transforms(config)
clip_duration = config.DATA.CLIP_DURATION
num_labels = config.MODEL.HEAD.NUM_CLASSES

def get_clip_tensors(video):
    """Get all clip (chunk) tensors from a video.
        Each last for 'clip_duration' seconds."""
    video_duration = float(video._duration)
    all_clip_tensors = []

    stride = STRIDE if STRIDE is not None else clip_duration

    for clip_start_sec in np.arange(0, video_duration, stride):
        clip_end_sec = min(clip_start_sec + clip_duration, video_duration)
        clip_data = video.get_clip(start_sec=clip_start_sec, end_sec=clip_end_sec)
        if clip_data['video'] is None:
            continue
        clip_data = transform(clip_data)
        clip_tensor = clip_data['video']  # (C, T, H, W)
        all_clip_tensors.append(clip_tensor)
    return all_clip_tensors

def get_true_label_array(sample):
    '''Get the ground truth label array for a sample.'''
    label_arr = np.zeros((num_labels, ))
    if pd.notnull(sample.actions):
        actions = sample.actions.split(";")
        for gt_action in actions:
            current_action, start_time, end_time = gt_action.split(' ')
            label_ind = action_id2label[current_action]
            label_arr[label_ind] = 1
    return label_arr

class InferDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        vid_path = f"{VIDEO_DIRS}/{sample['id']}.mp4"
        video = EncodedVideo.from_path(vid_path, decode_audio=False)
        ground_truth = get_true_label_array(sample)

        # concat clip tensors at the time dimension to write to a gif
        all_clip_tensors = get_clip_tensors(video)
        # Prepare video tensor for inference
        inp_video_tensors = torch.stack(all_clip_tensors, dim=0)  # (B, C, T, H, W)
        inp_video_tensors = inp_video_tensors.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) -> (B, T, C, H, W)

        return inp_video_tensors, ground_truth

def inference(lit_module, df):
    """Inference on the dataset. In each iteration, the model predicts the labels for a video."""
    list_gt_labels = []
    list_pred_labels = []
    dataset = InferDataset(df)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                            num_workers=NUM_PROCESSES)

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        inp_video_tensors , gt_labels = batch
        inp_video_tensors = inp_video_tensors[0]
        gt_labels = gt_labels[0]
        with torch.no_grad():
            inp_video_tensors = inp_video_tensors.to(DEVICE)
            model_output = lit_module(inp_video_tensors).sigmoid().cpu().numpy()
            model_output = model_output.max(axis=0)
            list_pred_labels.append(model_output)
            list_gt_labels.append(gt_labels)

    all_gt_labels = np.stack(list_gt_labels)
    all_pred_labels = np.stack(list_pred_labels)

    return all_gt_labels, all_pred_labels


all_gt_labels, all_pred_labels = inference(lit_module, df)
print('Model mAP:', compute_multilabel_mAP(all_gt_labels, all_pred_labels, num_labels))