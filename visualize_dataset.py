import argparse

from fvcore.common.config import CfgNode

from src.datasets import create_dataset
from src.utils.visualizations import investigate_video, display_gif

parser = argparse.ArgumentParser(description="Train a video model")
parser.add_argument("--config", help="The config file", 
                        default="src/config/cls_svt_ucf101_s224_f8_exp0.yaml")
args = parser.parse_args()

def vis():
    """Visualize one sample of training data"""
    config = CfgNode.load_yaml_with_base(args.config)
    config = CfgNode(config)

    dataset = create_dataset(config)
    train_ds = dataset.get_train_dataset()
    sample_video = next(iter(train_ds))
    print(sample_video.keys())
    investigate_video(sample_video, dataset.get_id2label())
    print(len(dataset.get_id2label()))

    video_tensor = sample_video["video"]
    save_path =  "assets/sample_ufc101.gif"
    print("Save sample to:", save_path)
    display_gif(video_tensor, save_path, config.DATA.MEAN, config.DATA.STD)

if __name__ == '__main__':
    vis()
