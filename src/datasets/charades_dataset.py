import os
import pathlib
from typing import Dict

from torch.utils.data import Dataset
from fvcore.common.config import CfgNode
import pytorchvideo.data
import pandas as pd

import csv
import functools
import itertools
import os
from collections import defaultdict
from typing import Any, Callable, List, Optional, Tuple, Type

import torch
import torch.utils.data

from iopath.common.file_io import g_pathmgr

from transformers import AutoTokenizer

from pytorchvideo.data import Charades
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.frame_video import FrameVideo
from pytorchvideo.data.utils import MultiProcessSampler


from .dataset_abstract import DatasetAbstract
from .transformations import get_train_transforms, get_val_transforms


# Just in case we want to customize the Charades dataset, now not needed
class CustomCharadesForCaptioning(Charades):
    def __init__(self,
        data_path: str,
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        video_path_prefix: str = "",
        frames_per_clip: Optional[int] = None,
        tokenizer: AutoTokenizer = None,
        max_tokens=128) -> None:
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
            tokenizer (transformers.Tokenizer): a tokenizer to encode text into list of tokens
            max_tokens (int): max number of tokens, truncate if exceeds
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

        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    @property
    def video_sampler(self) -> torch.utils.data.Sampler:
        return self._video_sampler

    def __next__(self) -> dict:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                }
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        if self._loaded_video:
            video, video_index = self._loaded_video
            
        else:
            video_index = next(self._video_sampler_iter)
            path_to_video_frames = self._path_to_videos[video_index]
            video = FrameVideo.from_frame_paths(path_to_video_frames)
            self._loaded_video = (video, video_index)
            
        clip_start, clip_end, clip_index, aug_index, is_last_clip = self._clip_sampler(
            self._next_clip_start_time, video.duration, {}
        )
        # Only load the clip once and reuse previously stored clip if there are multiple
        # views for augmentations to perform on the same clip.
        if aug_index == 0:
            self._loaded_clip = video.get_clip(clip_start, clip_end, self._frame_filter)

        frames, frame_indices = (
            self._loaded_clip["video"],
            self._loaded_clip["frame_indices"],
        )
        self._next_clip_start_time = clip_end

        if is_last_clip:
            self._loaded_video = None
            self._next_clip_start_time = 0.0

        # Merge unique labels from each frame into clip label.
       
        label = self._labels[video_index]
        encoded_label = self.tokenizer(label, max_length=self.max_tokens, padding='max_length', 
                                              truncation=True, return_tensors='pt')

        sample_dict = {
            "video": frames,
            "label": encoded_label,
            "label_str": self._video_labels[video_index],
            "video_name": str(video_index),
            "video_index": video_index,
            "clip_index": clip_index,
            "aug_index": aug_index,
        }
        if self._transform is not None:
            sample_dict = self._transform(sample_dict)

        return sample_dict

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
        labels = defaultdict(set)
        with g_pathmgr.open(video_path_label_file, "r") as f:

            # Space separated CSV with format: original_vido_id video_id frame_id path labels
            csv_reader = csv.DictReader(f, delimiter=" ")
            for row in csv_reader:
                assert len(row) == 5
                video_name = row["original_vido_id"]
                path = os.path.join(prefix, row["path"])
                image_paths[video_name].append(path)
                frame_cap = row["caption"]
                labels[video_name].add(frame_cap)

        # Extract image paths from dictionary and return paths and labels as list.
        video_names = image_paths.keys()
        image_paths = [image_paths[key] for key in video_names]
        labels = [list(labels[key])[0] for key in video_names]
        # print(video_names[0])
        # Aggregate labels from all frames to form video-level labels.
        # video_labels = [list(set(itertools.chain(*label_list))) for label_list in labels]
        return image_paths, labels, labels

class CustomCharadesForActionClassification(Charades):
    def __init__(self,
        data_path: str,
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        video_path_prefix: str = "",
        frames_per_clip: Optional[int] = None,
        fps:float=1.5) -> None:
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
            fps: video's frame per second
        """

        super().__init__(data_path,
                        clip_sampler,
                        video_sampler,
                        transform,
                        video_path_prefix,
                        frames_per_clip)
        self.fps = fps

    def __next__(self) -> dict:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                }
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        if self._loaded_video:
            video, video_index = self._loaded_video
        else:
            video_index = next(self._video_sampler_iter)
            path_to_video_frames = self._path_to_videos[video_index]
            video = FrameVideo.from_frame_paths(path_to_video_frames, fps=self.fps)
            self._loaded_video = (video, video_index)

        clip_start, clip_end, clip_index, aug_index, is_last_clip = self._clip_sampler(
            self._next_clip_start_time, video.duration, {}
        )
        # Only load the clip once and reuse previously stored clip if there are multiple
        # views for augmentations to perform on the same clip.
        if aug_index == 0:
            self._loaded_clip = video.get_clip(clip_start, clip_end, self._frame_filter)

        frames, frame_indices = (
            self._loaded_clip["video"],
            self._loaded_clip["frame_indices"],
        )
        self._next_clip_start_time = clip_end

        if is_last_clip:
            self._loaded_video = None
            self._next_clip_start_time = 0.0

        # Merge unique labels from each frame into clip label.
        labels_by_frame = [
            self._labels[video_index][i]
            for i in range(min(frame_indices), max(frame_indices) + 1)
        ]
        clip_label = list(set(itertools.chain.from_iterable(labels_by_frame)))
        sample_dict = {
            "video": frames,
            "label": labels_by_frame,
            "clip_label": clip_label,
            "video_label": self._video_labels[video_index],
            "video_name": str(video_index),
            "video_index": video_index,
            "clip_index": clip_index,
            "aug_index": aug_index,
        }
        if self._transform is not None:
            sample_dict = self._transform(sample_dict)

        return sample_dict

class CharadesCaptionDataset(DatasetAbstract):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.config = config
        self.dataset_root_path = pathlib.Path(config.DATA.ROOT_PATH)
        self.train_csv_path = self.dataset_root_path / config.DATA.TRAIN_CSV
        self.test_csv_path = self.dataset_root_path / config.DATA.TEST_CSV

        self.clip_duration = config.DATA.CLIP_DURATION

        self.train_transforms = get_train_transforms(config)
        self.val_transforms = get_val_transforms(config)

        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL.HEAD.LANGUAGE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_train_dataset(self) -> Dataset:
        clip_sampler = pytorchvideo.data.make_clip_sampler("random", self.clip_duration)
        train_dataset = CustomCharadesForCaptioning(
                            data_path=self.train_csv_path, 
                            clip_sampler=clip_sampler,
                            transform=self.train_transforms,
                            tokenizer=self.tokenizer)
        return train_dataset

    def get_val_dataset(self) -> Dataset:
        clip_sampler = pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration)
        val_dataset = CustomCharadesForCaptioning(
                            data_path=self.test_csv_path, 
                            clip_sampler=clip_sampler,
                            transform=self.val_transforms,
                            tokenizer=self.tokenizer)
        return val_dataset


class CharadesActionClassification(DatasetAbstract):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.config = config
        self.dataset_root_path = pathlib.Path(config.DATA.ROOT_PATH)
        self.train_csv_path = self.dataset_root_path / config.DATA.TRAIN_CSV
        self.test_csv_path = self.dataset_root_path / config.DATA.TEST_CSV

        self.label_map = pd.read_csv(self.dataset_root_path / "Charades_v1_classes_new_map.csv")

        self.label2id = self.label_map.set_index('action')['label'].to_dict()
        self.id2label = self.label_map.set_index('label')['action'].to_dict()

        self.clip_duration = config.DATA.CLIP_DURATION
        self.fps = config.DATA.FPS

        self.train_transforms = get_train_transforms(config)
        self.val_transforms = get_val_transforms(config)

    def get_train_dataset(self) -> Dataset:
        train_dataset = CustomCharadesForActionClassification(
            data_path=str(self.train_csv_path),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
            transform=self.train_transforms,
            fps=self.fps
        )
        return train_dataset

    def get_val_dataset(self) -> Dataset:
        val_dataset = CustomCharadesForActionClassification(
            data_path=str(self.test_csv_path),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
            transform=self.val_transforms,
            fps=self.fps
        )
        return val_dataset

    def get_id2label(self) -> Dict:
        return self.id2label
