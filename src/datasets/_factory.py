from fvcore.common.config import CfgNode

from .dataset_abstract import DatasetAbstract
from .ucf101_dataset import UFC101Dataset
from .charades_caption_dataset import CharadesCaptionDataset
    
def create_dataset(config: CfgNode) -> DatasetAbstract:
    dataset_name = config.DATA.DATASET
    if dataset_name == 'ucf101':
        return UFC101Dataset(config)
    elif dataset_name == 'charades_caption':
        return CharadesCaptionDataset(config)
    
    raise NotImplementedError(f'Dataset not implemented:{dataset_name}')