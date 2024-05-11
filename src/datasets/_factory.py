
from .dataset_abstract import DatasetAbstract
from .ucf101_dataset import UFC101Dataset

    
def create_dataset(config) -> DatasetAbstract:
    dataset_name = config.DATA.DATASET
    if dataset_name == 'ucf101':
        return UFC101Dataset(config)
    
    raise NotImplementedError(f'Dataset not implemented:{dataset_name}')