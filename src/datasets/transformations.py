import os

from fvcore.common.config import CfgNode

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

def get_train_transforms(config: CfgNode):
    mean = config.DATA.MEAN
    std = config.DATA.STD
    resize_to = (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
    if config.MODEL.TYPE == "captioning":
        train_transforms = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(config.DATA.NUM_SAMPLED_FRAMES_MULT*config.DATA.NUM_SAMPLED_FRAMES),
                            Resize(resize_to),
                            RandomHorizontalFlip(p=0.5),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                        ]
                    ),
                ),
            ]
        )
    else:
        train_transforms = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(config.DATA.NUM_SAMPLED_FRAMES),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            Resize(resize_to),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )
    return train_transforms


def get_val_transforms(config: CfgNode):
    resize_to = (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
    mean = config.DATA.MEAN
    std = config.DATA.STD
    if config.MODEL.TYPE == "captioning":
        val_transforms = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(
                                config.DATA.NUM_SAMPLED_FRAMES_MULT * config.DATA.NUM_SAMPLED_FRAMES),
                            Resize(resize_to),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                        ]
                    ),
                ),
            ]
        )
    else:
        val_transforms = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(config.DATA.NUM_SAMPLED_FRAMES),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            Resize(resize_to),
                        ]
                    ),
                ),
            ]
        )
    return val_transforms

