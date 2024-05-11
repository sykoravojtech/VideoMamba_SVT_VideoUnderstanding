import torch
from fvcore.common.config import CfgNode

from src.models import create_model
from src.datasets import create_dataset
from src.utils.visualizations import investigate_video, display_gif

CONFIG_FILE = 'src/config/cls_svt_s224_f8.yaml'

def train():
    # config = CfgNode()
    config = CfgNode.load_yaml_with_base(CONFIG_FILE)
    config = CfgNode(config)

    dataset = create_dataset(config)
    train_ds = dataset.get_train_dataset()
    sample_video = next(iter(train_ds))
    print(sample_video.keys())
    investigate_video(sample_video, dataset.get_id2label())
    print(len(dataset.get_id2label()))

    video_tensor = sample_video["video"]
    display_gif(video_tensor, "assets/sample_ufc101.gif", config.DATA.MEAN, config.DATA.STD)

if __name__ == '__main__':
    train()
