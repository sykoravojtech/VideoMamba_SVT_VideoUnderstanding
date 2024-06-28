# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr

import pytorchvideo
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset

from .dataset_abstract import DatasetAbstract
from .transformations import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)

ID2LABEL = {0: 'brush_hair', 1: 'cartwheel', 2: 'catch', 3: 'chew', 4: 'clap', 5: 'climb', 6: 'climb_stairs', 
            7: 'dive', 8: 'draw_sword', 9: 'dribble', 10: 'drink', 11: 'eat', 12: 'fall_floor', 
            13: 'fencing', 14: 'flic_flac', 15: 'golf', 16: 'handstand', 17: 'hit', 18: 'hug',
            19: 'jump', 20: 'kick', 21: 'kick_ball', 22: 'kiss', 23: 'laugh', 24: 'pick',
            25: 'pour', 26: 'pullup', 27: 'punch', 28: 'push', 29: 'pushup', 30: 'ride_bike', 
            31: 'ride_horse', 32: 'run', 33: 'shake_hands', 34: 'shoot_ball', 35: 'shoot_bow', 
            36: 'shoot_gun', 37: 'sit', 38: 'situp', 39: 'smile', 40: 'smoke', 41: 'somersault', 
            42: 'stand', 43: 'swing_baseball', 44: 'sword', 45: 'sword_exercise', 46: 'talk', 
            47: 'throw', 48: 'turn', 49: 'walk', 50: 'wave'}
LABEL2ID = {v:k for k,v in ID2LABEL.items()}

class Hmdb51LabeledVideoPaths:
    """
    Pre-processor for Hmbd51 dataset mentioned here -
        https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/

    This dataset consists of classwise folds with each class consisting of 3
        folds (splits).

    The videos directory is of the format,
        video_dir_path/class_x/<somevideo_name>.avi
        ...
        video_dir_path/class_y/<somevideo_name>.avi

    The splits/fold directory is of the format,
        folds_dir_path/class_x_test_split_1.txt
        folds_dir_path/class_x_test_split_2.txt
        folds_dir_path/class_x_test_split_3.txt
        ...
        folds_dir_path/class_y_test_split_1.txt
        folds_dir_path/class_y_test_split_2.txt
        folds_dir_path/class_y_test_split_3.txt

    And each text file in the splits directory class_x_test_split_<1 or 2 or 3>.txt
        <a video as in video_dir_path/class_x> <0 or 1 or 2>
        where 0,1,2 corresponds to unused, train split respectively.

    Each video has name of format
        <some_name>_<tag1>_<tag2>_<tag_3>_<tag4>_<tag5>_<some_id>.avi
    For more details on tags -
        https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
    """

    _allowed_splits = [1, 2, 3]
    _split_type_dict = {"train": 1, "test": 2, "unused": 0}

    @classmethod
    def from_dir(
        cls, data_path: str, split_id: int = 1, split_type: str = "train"
    ) -> Hmdb51LabeledVideoPaths:
        """
        Factory function that creates Hmdb51LabeledVideoPaths object form a splits/folds
        directory.

        Args:
            data_path (str): The path to the splits/folds directory of HMDB51.
            split_id (int): Fold id to be loaded. Belongs to [1,2,3]
            split_type (str): Split/Fold type to be loaded. It belongs to one of the
                following,
                - "train"
                - "test"
                - "unused" (This is a small set of videos that are neither
                of part of test or train fold.)
        """
        print('from dir')
        data_path = pathlib.Path(data_path)
        if not data_path.is_dir():
            return RuntimeError(f"{data_path} not found or is not a directory.")
        if not int(split_id) in cls._allowed_splits:
            return RuntimeError(
                f"{split_id} not found in allowed split id's {cls._allowed_splits}."
            )
        file_name_format = "_test_split" + str(int(split_id))
        file_paths = sorted(
            (
                f
                for f in data_path.iterdir()
                if f.is_file() and f.suffix == ".txt" and file_name_format in f.stem
            )
        )
        return cls.from_csvs(file_paths, split_type)

    @classmethod
    def from_csvs(
        cls, file_paths: List[Union[pathlib.Path, str]], split_type: str = "train"
    ) -> Hmdb51LabeledVideoPaths:
        """
        Factory function that creates Hmdb51LabeledVideoPaths object form a list of
        split files of .txt type

        Args:
            file_paths (List[Union[pathlib.Path, str]]) : The path to the splits/folds
                    directory of HMDB51.
            split_type (str): Split/Fold type to be loaded.
                - "train"
                - "test"
                - "unused"
        """
        video_paths_and_label = []
        for file_path in file_paths:
            file_path = pathlib.Path(file_path)
            assert g_pathmgr.exists(file_path), f"{file_path} not found."
            if not (file_path.suffix == ".txt" and "_test_split" in file_path.stem):
                return RuntimeError(f"Ivalid file: {file_path}")

            action_name = "_"
            action_name = action_name.join((file_path.stem).split("_")[:-2])
            with g_pathmgr.open(file_path, "r") as f:
                for path_label in f.read().splitlines():
                    line_split = path_label.rsplit(None, 1)

                    if not int(line_split[1]) == cls._split_type_dict[split_type]:
                        continue

                    file_path = os.path.join(action_name, line_split[0])
                    meta_tags = line_split[0].split("_")[-6:-1]
                    video_paths_and_label.append(
                        (file_path, {"label": LABEL2ID[action_name], "label_str": action_name, "meta_tags": meta_tags})
                    )

        assert (
            len(video_paths_and_label) > 0
        ), f"Failed to load dataset from {file_path}."
        return cls(video_paths_and_label)

    def __init__(
        self, paths_and_labels: List[Tuple[str, Optional[dict]]], path_prefix=""
    ) -> None:
        """
        Args:
            paths_and_labels [(str, int)]: a list of tuples containing the video
                path and integer label.
        """
        self._paths_and_labels = paths_and_labels
        self._path_prefix = path_prefix

    def path_prefix(self, prefix):
        self._path_prefix = prefix

    path_prefix = property(None, path_prefix)

    def __getitem__(self, index: int) -> Tuple[str, dict]:
        """
        Args:
            index (int): the path and label index.

        Returns:
            The path and label tuple for the given index.
        """
        path, label = self._paths_and_labels[index]
        return (os.path.join(self._path_prefix, path), label)

    def __len__(self) -> int:
        """
        Returns:
            The number of video paths and label pairs.
        """
        return len(self._paths_and_labels)


def Hmdb51(
    data_path: pathlib.Path,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[dict], Any]] = None,
    video_path_prefix: str = "",
    split_id: int = 1,
    split_type: str = "train",
    decode_audio=True,
    decoder: str = "pyav",
) -> LabeledVideoDataset:
    """
    A helper function to create ``LabeledVideoDataset`` object for HMDB51 dataset

    Args:
        data_path (pathlib.Path): Path to the data. The path type defines how the data
            should be read:

            * For a file path, the file is read and each line is parsed into a
              video path and label.
            * For a directory, the directory structure defines the classes
              (i.e. each subdirectory is a class).

        clip_sampler (ClipSampler): Defines how clips should be sampled from each
            video. See the clip sampling documentation for more information.

        video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
            video container. This defines the order videos are decoded and,
            if necessary, the distributed split.

        transform (Callable): This callable is evaluated on the clip output before
            the clip is returned. It can be used for user defined preprocessing and
            augmentations to the clips. See the ``LabeledVideoDataset`` class for
            clip output format.

        video_path_prefix (str): Path to root directory with the videos that are
            loaded in LabeledVideoDataset. All the video paths before loading
            are prefixed with this path.

        split_id (int): Fold id to be loaded. Options are 1, 2 or 3

        split_type (str): Split/Fold type to be loaded. Options are ("train", "test" or
            "unused")

        decoder (str): Defines which backend should be used to decode videos.
    """

    torch._C._log_api_usage_once("PYTORCHVIDEO.dataset.Hmdb51")

    labeled_video_paths = Hmdb51LabeledVideoPaths.from_dir(
        data_path, split_id=split_id, split_type=split_type
    )
    labeled_video_paths.path_prefix = video_path_prefix
    dataset = LabeledVideoDataset(
        labeled_video_paths,
        clip_sampler,
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )

    return dataset

class HMDB51Dataset(DatasetAbstract):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.config = config
        self.dataset_root_path = pathlib.Path(config.DATA.ROOT_PATH)
        self.fold_split_paths = self.dataset_root_path / "testTrainMulti_7030_splits"
        self.videos_path = self.dataset_root_path / "HMDB51_videos"
        self.fold = config.DATA.FOLD
        self.label2id = LABEL2ID
        self.id2label = ID2LABEL

        self.clip_duration = config.DATA.CLIP_DURATION

        self.train_transforms = get_train_transforms(config)
        self.val_transforms = get_val_transforms(config)


    def get_train_dataset(self) -> Dataset:
        train_dataset = Hmdb51(
            data_path=self.fold_split_paths, 
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.clip_duration),
            video_path_prefix=self.videos_path,
            split_id=self.fold,
            split_type="train",
            decode_audio=False,
            transform=self.train_transforms,
            decoder='decord'
        )
        return train_dataset

    def get_val_dataset(self) -> Dataset:
        val_dataset = Hmdb51(
            data_path=self.fold_split_paths,
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
            video_path_prefix=self.videos_path,
            split_id=self.fold,
            split_type="test",
            decode_audio=False,
            transform=self.val_transforms,
            decoder='decord'
        )   
        return val_dataset

    def get_id2label(self) -> Dict:
        return self.id2label