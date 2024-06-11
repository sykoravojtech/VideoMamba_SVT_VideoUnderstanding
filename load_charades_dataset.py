
import torch
import pytorchvideo
from pytorchvideo.data import Charades

import torch
import torch.utils.data
from src.datasets.transformations import get_train_transforms, get_val_transforms
from fvcore.common.config import CfgNode

from src.utils.visualizations import investigate_video, display_gif

FOLDER = 'data'
PROCESSED_CSV_PATH = f'{FOLDER}/processed/Charades/Charades_v1_train_pytorchvideo_perframe.csv'

config = CfgNode.load_yaml_with_base("src/config/cls_svt_ucf101_s224_f8_exp0.yaml")
config = CfgNode(config)

clip_duration = 3 # Hard code to reduce the duration of the clip
clip_sampler = pytorchvideo.data.make_clip_sampler("random", clip_duration)

### Create the dataset using the per-frame annotations
ds = Charades(data_path=PROCESSED_CSV_PATH, 
                    clip_sampler=clip_sampler,
                    transform=get_val_transforms(config),)

sample_video = next(iter(ds))
print(sample_video.keys())

video_tensor = sample_video["video"]
print(video_tensor)
save_path =  "assets/sample_charades.gif"
print("Save sample to:", save_path)
display_gif(video_tensor, save_path, config.DATA.MEAN, config.DATA.STD)









