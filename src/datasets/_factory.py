from fvcore.common.config import CfgNode

from .dataset_abstract import DatasetAbstract
from .ucf101_dataset import UFC101Dataset
from .charades_dataset import CharadesActionClassification

    
def create_dataset(config: CfgNode) -> DatasetAbstract:
    dataset_name = config.DATA.DATASET
    task = config.MODEL.TYPE
    if dataset_name == 'ucf101':
        return UFC101Dataset(config)
    if dataset_name == 'charades' and task == 'classification':
        return CharadesActionClassification(config)
    
    raise NotImplementedError(f'Dataset not implemented:{dataset_name}')