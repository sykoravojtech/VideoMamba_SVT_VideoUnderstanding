import os
import pathlib
from typing import Dict

from torch.utils.data import Dataset
from fvcore.common.config import CfgNode
import pytorchvideo.data

from .dataset_abstract import DatasetAbstract
from .transformations import get_train_transforms, get_val_transforms

class CharadesDataset(DatasetAbstract):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.config = config
        self.dataset_root_path = pathlib.Path(config.DATA.ROOT_PATH)
        self.train_csv_path = self.dataset_root_path / "Charades_v1_train.csv"  # TODO: Adjust csv files to match pytorchvideo specs
        self.test_csv_path = self.dataset_root_path / "Charades_v1_test.csv"  # TODO: Adjust csv files to match pytorchvideo specs
        self.videos_path = self.dataset_root_path / "videos"  # Given Charades' structure, videos are not split.
        
        self.class_labels = list(range(pytorchvideo.data.Charades.NUM_CLASSES))  # Charades has 157 classes
        self.label2id = {label: i for i, label in enumerate(self.class_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        self.clip_duration = config.DATA.NUM_SAMPLED_FRAMES * config.DATA.SAMPLE_RATE / config.DATA.FPS

        self.train_transforms = get_train_transforms(config)
        self.val_transforms = get_val_transforms(config)

    def get_train_dataset(self) -> Dataset:
        train_dataset = pytorchvideo.data.Charades(
            data_path=str(self.train_csv_path),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.clip_duration),
            video_path_prefix=str(self.videos_path),
            transform=self.train_transforms,
        )
        return train_dataset

    def get_val_dataset(self) -> Dataset:
        val_dataset = pytorchvideo.data.Charades(
            data_path=str(self.test_csv_path),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
            video_path_prefix=str(self.videos_path),
            transform=self.val_transforms,
        )
        return val_dataset

    def get_id2label(self) -> Dict:
        return self.id2label
