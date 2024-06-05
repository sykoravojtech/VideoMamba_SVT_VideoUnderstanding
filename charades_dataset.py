# %% [markdown]
# 

# %%
import pandas as pd
import numpy as np
import os

# %%
FOLDER = 'data'

# %%
df = pd.read_csv(f'{FOLDER}/raw/Charades/Charades_v1_train.csv')

# %%
df.head(3)

# %%
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
action_df.to_csv(f'{FOLDER}/processed/Charades_v1_classes.csv', index=False)

# %%
# original_vido_id video_id frame_id path labels

# %%
action_id_to_label = action_df.set_index('action_id')['label'].to_dict()

# %%
action_id_to_label

# %%
def get_label(str_labels):
    int_labels = []
    if pd.isnull(str_labels):
        return ''
    for item in str_labels.split(';'):
        action_id = item.split(' ')[0]
        int_labels.append(str(action_id_to_label[action_id]))
    return "," .join(int_labels)

# %%
VIDEO_FOLDER = f'{FOLDER}/raw/Charades/videos'
formated_df = pd.DataFrame()
formated_df['original_vido_id'] = df['id'].copy()
formated_df['video_id'] = 'None'
formated_df['frame_id'] = 'None'
formated_df['path'] = df['id'].map(lambda x: os.path.join(VIDEO_FOLDER, x + '.mp4'))
formated_df['labels'] = df['actions'].map(get_label)

PROCESSED_CSV_PATH = f'{FOLDER}/processed/Charades/Charades_v1_train_pytorchvideo.csv'
formated_df.to_csv(PROCESSED_CSV_PATH, index=False, sep=' ')

# %%
formated_df.head()

# %%
import torch
import pytorchvideo
from pytorchvideo.data import Charades

# %%
import csv
import functools
import itertools
import os
from collections import defaultdict
from typing import Any, Callable, List, Optional, Tuple, Type

import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.frame_video import FrameVideo

from pytorchvideo.data.utils import MultiProcessSampler

class CustomCharades(Charades):
    def __init__(self,
        data_path: str,
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        video_path_prefix: str = "",
        frames_per_clip: Optional[int] = None,) -> None:
        """
        Args:
            data_path (str): Path to the data file. This file must be a space
                separated csv with the format: (original_vido_id video_id frame_id
                path_labels)

            clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

            video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

            transform (Optional[Callable]): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations on the clips. The clip output format is described in __next__().

            video_path_prefix (str): prefix path to add to all paths from data_path.

            frames_per_clip (Optional[int]): The number of frames per clip to sample.
        """

        torch._C._log_api_usage_once("PYTORCHVIDEO.dataset.Charades.__init__")

        self._transform = transform
        self._clip_sampler = clip_sampler
        (
            self._path_to_videos,
            self._labels,
            self._video_labels,
        ) = self._read_video_paths_and_labels(data_path, prefix=video_path_prefix)
        self._video_sampler = video_sampler(self._path_to_videos)
        self._video_sampler_iter = None  # Initialized on first call to self.__next__()
        self._frame_filter = (
            functools.partial(
                Charades._sample_clip_frames,
                frames_per_clip=frames_per_clip,
            )
            if frames_per_clip is not None
            else None
        )

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video = None
        self._loaded_clip = None
        self._next_clip_start_time = 0.0

    def _read_video_paths_and_labels(self,
        video_path_label_file: List[str], prefix: str = ""
    ) -> Tuple[List[str], List[int]]:
        """
        Args:
            video_path_label_file (List[str]): a file that contains frame paths for each
                video and the corresponding frame label. The file must be a space separated
                csv of the format:
                    `original_vido_id video_id frame_id path labels`

            prefix (str): prefix path to add to all paths from video_path_label_file.

        """
        image_paths = defaultdict(list)
        labels = defaultdict(list)
        with g_pathmgr.open(video_path_label_file, "r") as f:

            # Space separated CSV with format: original_vido_id video_id frame_id path labels
            csv_reader = csv.DictReader(f, delimiter=" ")
            for row in csv_reader:
                assert len(row) == 5
                video_name = row["original_vido_id"]
                path = os.path.join(prefix, row["path"])
                image_paths[video_name].append(path)
                frame_labels = row["labels"].replace('"', "")
                label_list = []
                if frame_labels:
                    label_list = [int(x) for x in frame_labels.split(",")]

                labels[video_name].append(label_list)

        # Extract image paths from dictionary and return paths and labels as list.
        video_names = image_paths.keys()
        image_paths = [image_paths[key] for key in video_names]
        labels = [labels[key] for key in video_names]
        # print(video_names[0])
        # Aggregate labels from all frames to form video-level labels.
        video_labels = [list(set(itertools.chain(*label_list))) for label_list in labels]
        return image_paths, labels, video_labels

# %%
clip_duration = 8 * 4 / 30
clip_sampler = pytorchvideo.data.make_clip_sampler("random", clip_duration)
ds = CustomCharades(data_path=PROCESSED_CSV_PATH, clip_sampler=clip_sampler)

# %%
iter(ds)

# %%
sample_video = next(iter(ds))
# print(sample_video.keys())
# investigate_video(sample_video, dataset.get_id2label())
# print(len(dataset.get_id2label()))

# video_tensor = sample_video["video"]
# save_path =  "assets/sample_ufc101.gif"
# print("Save sample to:", save_path)
# display_gif(video_tensor, save_path, config.DATA.MEAN, config.DATA.STD)


# %%


# %%



