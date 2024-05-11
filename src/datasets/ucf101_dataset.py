import os
import pathlib
from typing import Dict

from torch.utils.data import Dataset
from fvcore.common.config import CfgNode
import pytorchvideo.data

from .dataset_abstract import DatasetAbstract
from .transformations import get_train_transforms, get_val_transforms

class UFC101Dataset(DatasetAbstract):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.config = config
        self.dataset_root_path = pathlib.Path(config.DATA.ROOT_PATH)
        self.all_video_file_paths = self.get_all_vid_paths(self.dataset_root_path)
        self.class_labels = sorted({str(path).split("/")[-2] for path in self.all_video_file_paths})
        # print(str(self.all_video_file_paths[0]).split("/"))
        self.label2id = {label: i for i, label in enumerate(self.class_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        self.clip_duration = config.DATA.NUM_SAMPLED_FRAMES * config.DATA.SAMPLE_RATE / config.DATA.FPS

        self.train_transforms = get_train_transforms(config)
        self.val_transforms = get_val_transforms(config)


    def get_all_vid_paths(self, dataset_root_path: pathlib.Path):
        dataset_root_path = pathlib.Path(dataset_root_path)
        video_count_train = len(list(dataset_root_path.glob("train/*/*.avi")))
        video_count_val = len(list(dataset_root_path.glob("val/*/*.avi")))
        video_count_test = len(list(dataset_root_path.glob("test/*/*.avi")))
        video_total = video_count_train + video_count_val + video_count_test
        print(f"Total videos (train, val, test): {video_total}")

        all_video_file_paths = (
            list(dataset_root_path.glob("train/*/*.avi"))
            + list(dataset_root_path.glob("val/*/*.avi"))
            + list(dataset_root_path.glob("test/*/*.avi"))
        )

        return all_video_file_paths

    def get_train_dataset(self) -> Dataset:
        train_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join(self.dataset_root_path, "train"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.clip_duration),
            decode_audio=False,
            transform=self.train_transforms,
        )
        return train_dataset

    def get_val_dataset(self) -> Dataset:
        val_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join(self.dataset_root_path, "val"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
            decode_audio=False,
            transform=self.val_transforms,
        )   
        return val_dataset

    def get_id2label(self) -> Dict:
        return self.id2label