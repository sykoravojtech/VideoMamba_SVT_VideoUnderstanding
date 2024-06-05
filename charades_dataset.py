
import pandas as pd
import numpy as np
import os


FOLDER = 'data'


df = pd.read_csv(f'{FOLDER}/raw/Charades/Charades_v1_train.csv')


df.head(3)


# OBJECT_TXT = f'{FOLDER}/raw/Charades/Charades_v1_objectclasses.txt'
# object_df = pd.read_csv(OBJECT_TXT, sep=' ', header=None)
# object_df.columns = ['object_id', 'object_name']
# object_df['object_name'] = object_df['object_name'].fillna('None')
# object_df['label'] = np.arange(len(object_df))
# object_df.to_csv(f'{FOLDER}/processed/Charades_v1_objectclasses.csv', index=False)

ACTIONT_TXT = f'{FOLDER}/raw/Charades/Charades_v1_classes.txt'
action_ids = []
actions = []
for row in open(ACTIONT_TXT):
    row = row.split(' ')
    action_ids.append(row[0])
    actions.append(' '.join(row[1:]).strip())
    # break

action_df = pd.DataFrame({'action_id': action_ids, 'action': actions})
action_df['label'] = np.arange(len(action_df))
os.makedirs(f'{FOLDER}/processed/Charades', exist_ok=True)
action_df.to_csv(f'{FOLDER}/processed/Charades/Charades_v1_classes.csv', index=False)


# original_vido_id video_id frame_id path labels

action_id_to_label = action_df.set_index('action_id')['label'].to_dict()

def get_label(str_labels):
    int_labels = []
    if pd.isnull(str_labels):
        return ''
    for item in str_labels.split(';'):
        action_id = item.split(' ')[0]
        int_labels.append(str(action_id_to_label[action_id]))
    return "," .join(int_labels)


VIDEO_FOLDER = f'{FOLDER}/raw/Charades/videos'
formated_df = pd.DataFrame() # this is per video annotation
formated_df['original_vido_id'] = df['id'].copy()
formated_df['video_id'] = 'None'
formated_df['frame_id'] = 'None'
formated_df['path'] = df['id'].map(lambda x: os.path.join(VIDEO_FOLDER, x + '.mp4'))
formated_df['labels'] = df['actions'].map(get_label)

# PROCESSED_CSV_PATH = f'{FOLDER}/processed/Charades/Charades_v1_train_pytorchvideo.csv'
# formated_df.to_csv(PROCESSED_CSV_PATH, index=False, sep=' ')

formated_df.head()


# make demo annotation for 1 clip
# extract frames, save them and create per-frame annotation
import cv2
import os

# perframe_df = pd.DataFrame(columns=['original_vido_id', 'video_id', 'frame_id',	'path',	'labels'])
perframe_dict = []

for i, row in formated_df.iterrows():
    video_path = row.path

    # Path to the folder to save frames
    output_folder = f'{FOLDER}/processed/Charades/frames_data/'

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video capture
    cap = cv2.VideoCapture(video_path)

    # Read and save frames until the video ends
    frame_count = 0
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        if ret:
            # Save the frame as an image file
            frame_path = os.path.join(output_folder, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_path, frame)

            perframe_dict.append({'original_vido_id': row.original_vido_id, 'video_id': row.video_id, 
                                'frame_id': frame_count, 'path': frame_path, 'labels': row.labels})

            frame_count += 1
        else:
            break

    # Release the video capture
    cap.release()
    break

perframe_df = pd.DataFrame(perframe_dict)


PROCESSED_CSV_PATH = f'{FOLDER}/processed/Charades/Charades_v1_train_pytorchvideo_perframe.csv'
perframe_df.to_csv(PROCESSED_CSV_PATH, index=False, sep=' ')


import torch
import pytorchvideo
from pytorchvideo.data import Charades


import csv
import functools
import itertools
import os
from collections import defaultdict
from typing import Any, Callable, List, Optional, Tuple, Type

import torch
import torch.utils.data
# from iopath.common.file_io import g_pathmgr
# from pytorchvideo.data.clip_sampling import ClipSampler
# from pytorchvideo.data.frame_video import FrameVideo

# from pytorchvideo.data.utils import MultiProcessSampler

# # Just in case we want to customize the Charades dataset, now not needed
# class CustomCharades(Charades):
#     def __init__(self,
#         data_path: str,
#         clip_sampler: ClipSampler,
#         video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
#         transform: Optional[Callable[[dict], Any]] = None,
#         video_path_prefix: str = "",
#         frames_per_clip: Optional[int] = None,) -> None:
#         """
#         Args:
#             data_path (str): Path to the data file. This file must be a space
#                 separated csv with the format: (original_vido_id video_id frame_id
#                 path_labels)

#             clip_sampler (ClipSampler): Defines how clips should be sampled from each
#                 video. See the clip sampling documentation for more information.

#             video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
#                 video container. This defines the order videos are decoded and,
#                 if necessary, the distributed split.

#             transform (Optional[Callable]): This callable is evaluated on the clip output before
#                 the clip is returned. It can be used for user defined preprocessing and
#                 augmentations on the clips. The clip output format is described in __next__().

#             video_path_prefix (str): prefix path to add to all paths from data_path.

#             frames_per_clip (Optional[int]): The number of frames per clip to sample.
#         """

#         torch._C._log_api_usage_once("PYTORCHVIDEO.dataset.Charades.__init__")

#         self._transform = transform
#         self._clip_sampler = clip_sampler
#         (
#             self._path_to_videos,
#             self._labels,
#             self._video_labels,
#         ) = self._read_video_paths_and_labels(data_path, prefix=video_path_prefix)
#         self._video_sampler = video_sampler(self._path_to_videos)
#         self._video_sampler_iter = None  # Initialized on first call to self.__next__()
#         self._frame_filter = (
#             functools.partial(
#                 Charades._sample_clip_frames,
#                 frames_per_clip=frames_per_clip,
#             )
#             if frames_per_clip is not None
#             else None
#         )

#         # Depending on the clip sampler type, we may want to sample multiple clips
#         # from one video. In that case, we keep the store video, label and previous sampled
#         # clip time in these variables.
#         self._loaded_video = None
#         self._loaded_clip = None
#         self._next_clip_start_time = 0.0

#     @property
#     def video_sampler(self) -> torch.utils.data.Sampler:
#         return self._video_sampler

#     def __next__(self) -> dict:
#         """
#         Retrieves the next clip based on the clip sampling strategy and video sampler.

#         Returns:
#             A dictionary with the following format.

#             .. code-block:: text

#                 {
#                     'video': <video_tensor>,
#                     'label': <index_label>,
#                     'video_label': <index_label>
#                     'video_index': <video_index>,
#                     'clip_index': <clip_index>,
#                     'aug_index': <aug_index>,
#                 }
#         """
#         if not self._video_sampler_iter:
#             # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
#             self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

#         if self._loaded_video:
#             video, video_index = self._loaded_video
            
#         else:
#             video_index = next(self._video_sampler_iter)
#             path_to_video_frames = self._path_to_videos[video_index]
#             video = FrameVideo.from_frame_paths(path_to_video_frames)
#             print(video)
#             self._loaded_video = (video, video_index)
            
#         clip_start, clip_end, clip_index, aug_index, is_last_clip = self._clip_sampler(
#             self._next_clip_start_time, video.duration, {}
#         )
#         # Only load the clip once and reuse previously stored clip if there are multiple
#         # views for augmentations to perform on the same clip.
#         if aug_index == 0:
#             print(clip_start, clip_end, self._frame_filter)
#             self._loaded_clip = video.get_clip(clip_start, clip_end, self._frame_filter)

#         frames, frame_indices = (
#             self._loaded_clip["video"],
#             self._loaded_clip["frame_indices"],
#         )
#         self._next_clip_start_time = clip_end

#         if is_last_clip:
#             self._loaded_video = None
#             self._next_clip_start_time = 0.0

#         # Merge unique labels from each frame into clip label.
#         labels_by_frame = [
#             self._labels[video_index][i]
#             for i in range(min(frame_indices), max(frame_indices) + 1)
#         ]
#         sample_dict = {
#             "video": frames,
#             "label": labels_by_frame,
#             "video_label": self._video_labels[video_index],
#             "video_name": str(video_index),
#             "video_index": video_index,
#             "clip_index": clip_index,
#             "aug_index": aug_index,
#         }
#         if self._transform is not None:
#             sample_dict = self._transform(sample_dict)

#         return sample_dict

#     def _read_video_paths_and_labels(self,
#         video_path_label_file: List[str], prefix: str = ""
#     ) -> Tuple[List[str], List[int]]:
#         """
#         Args:
#             video_path_label_file (List[str]): a file that contains frame paths for each
#                 video and the corresponding frame label. The file must be a space separated
#                 csv of the format:
#                     `original_vido_id video_id frame_id path labels`

#             prefix (str): prefix path to add to all paths from video_path_label_file.

#         """
#         image_paths = defaultdict(list)
#         labels = defaultdict(list)
#         with g_pathmgr.open(video_path_label_file, "r") as f:

#             # Space separated CSV with format: original_vido_id video_id frame_id path labels
#             csv_reader = csv.DictReader(f, delimiter=" ")
#             for row in csv_reader:
#                 assert len(row) == 5
#                 video_name = row["original_vido_id"]
#                 path = os.path.join(prefix, row["path"])
#                 image_paths[video_name].append(path)
#                 frame_labels = row["labels"].replace('"', "")
#                 label_list = []
#                 if frame_labels:
#                     label_list = [int(x) for x in frame_labels.split(",")]

#                 labels[video_name].append(label_list)

#         # Extract image paths from dictionary and return paths and labels as list.
#         video_names = image_paths.keys()
#         image_paths = [image_paths[key] for key in video_names]
#         labels = [labels[key] for key in video_names]
#         # print(video_names[0])
#         # Aggregate labels from all frames to form video-level labels.
#         video_labels = [list(set(itertools.chain(*label_list))) for label_list in labels]
#         return image_paths, labels, video_labels


# Try visualizing the 1 video (just like in the visualize_dataset.py)
from src.datasets.transformations import get_train_transforms, get_val_transforms
from fvcore.common.config import CfgNode

config = CfgNode.load_yaml_with_base("src/config/cls_svt_ucf101_s224_f8_exp0.yaml")
config = CfgNode(config)

clip_duration = 3 # Hard code to reduce the duration of the clip
clip_sampler = pytorchvideo.data.make_clip_sampler("random", clip_duration)
ds = Charades(data_path=PROCESSED_CSV_PATH, 
                    clip_sampler=clip_sampler,
                    transform=get_val_transforms(config),)

from src.utils.visualizations import investigate_video, display_gif

sample_video = next(iter(ds))
print(sample_video.keys())
# investigate_video(sample_video, ds.get_id2label())
# print(len(ds.get_id2label()))

video_tensor = sample_video["video"]
print(video_tensor)
save_path =  "assets/sample_charades.gif"
print("Save sample to:", save_path)
display_gif(video_tensor, save_path, config.DATA.MEAN, config.DATA.STD)



# import cv2 as cv
# cap = cv.VideoCapture(formated_df.iloc[0].path)
# fps = cap.get(cv.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
# frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
# duration = frame_count/fps

# print('fps = ' + str(fps))
# print('number of frames = ' + str(frame_count))
# print('duration (S) = ' + str(duration))
# minutes = int(duration/60)
# seconds = duration%60
# print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))
# cap.release()








