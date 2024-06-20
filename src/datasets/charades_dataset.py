import os
import pathlib
from typing import Dict

from torch.utils.data import Dataset
from fvcore.common.config import CfgNode
import pytorchvideo.data
import pandas as pd

from .dataset_abstract import DatasetAbstract
from .transformations import get_train_transforms, get_val_transforms

class CharadesActionClassification(DatasetAbstract):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.config = config
        self.dataset_root_path = pathlib.Path(config.DATA.ROOT_PATH)
        self.train_csv_path = self.dataset_root_path / "Charades_per-frame_annotations_train.csv"
        self.test_csv_path = self.dataset_root_path / "Charades_per-frame_annotations_test.csv"

        self.label_map = pd.read_csv(self.dataset_root_path / "Charades_v1_classes_new_map.csv")

        self.label2id = self.label_map.set_index('action')['label'].to_dict()
        self.id2label = self.label_map.set_index('label')['action'].to_dict()

        self.clip_duration = config.DATA.NUM_SAMPLED_FRAMES * config.DATA.SAMPLE_RATE / config.DATA.FPS

        self.train_transforms = get_train_transforms(config)
        self.val_transforms = get_val_transforms(config)

    def get_train_dataset(self) -> Dataset:
        train_dataset = pytorchvideo.data.Charades(
            data_path=str(self.train_csv_path),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
            transform=self.train_transforms,
        )
        return train_dataset

    def get_val_dataset(self) -> Dataset:
        val_dataset = pytorchvideo.data.Charades(
            data_path=str(self.test_csv_path),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
            transform=self.val_transforms,
        )
        return val_dataset

    def get_id2label(self) -> Dict:
        return self.id2label
