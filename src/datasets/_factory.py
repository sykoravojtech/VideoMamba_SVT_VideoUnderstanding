from fvcore.common.config import CfgNode

from .dataset_abstract import DatasetAbstract
from .ucf101_dataset import UFC101Dataset
from .charades_dataset import CharadesCaptionDataset, CharadesActionClassification
from .hmdb51_dataset import HMDB51Dataset

def create_dataset(config: CfgNode) -> DatasetAbstract:
    dataset_name = config.DATA.DATASET
    task = config.MODEL.TYPE
    if dataset_name == 'ucf101':
        return UFC101Dataset(config)
    elif dataset_name == 'charades_caption':
        return CharadesCaptionDataset(config)
    if dataset_name == 'charades_action_classification':
        return CharadesActionClassification(config)
    if dataset_name == 'hmdb51':
        return HMDB51Dataset(config)
    
    raise NotImplementedError(f'Dataset not implemented:{dataset_name}')