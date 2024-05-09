import yaml
from fvcore.common.config import CfgNode

from src.models import create_model

CONFIG_FILE = 'src/config/cls_svt_s224_f8.yml'

def train():
    config = CfgNode()
    config.load_yaml_with_base(CONFIG_FILE)
    print(config)
    # with open(CONFIG_FILE) as stream:
    #     try:
    #         config = yaml.safe_load(stream)
    #     except yaml.YAMLError as exc:
    #         print(exc)

    # model = create_model(config)
    # print(model)

if __name__ == '__main__':
    train()
