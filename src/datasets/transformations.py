import os


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

# mean = image_processor.image_mean
# std = image_processor.image_std



# if "shortest_edge" in image_processor.size:
#     height = width = image_processor.size["shortest_edge"]
# else:
#     height = image_processor.size["height"]
#     width = image_processor.size["width"]

def get_train_transforms(config):
    mean = config.DATA.MEAN
    std = config.DATA.STD
    resize_to = (config.DATA.TRAIN_CROP_SIZE, config.DATA.TRAIN_CROP_SIZE)
    train_transforms = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(config.DATA.NUM_SAMPLED_FRAMES),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(resize_to),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )
    return train_transforms


def get_val_transforms(config):
    resize_to = (config.DATA.TRAIN_CROP_SIZE, config.DATA.TRAIN_CROP_SIZE)
    mean = config.DATA.MEAN
    std = config.DATA.STD
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

