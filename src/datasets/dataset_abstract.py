import abc
from torch.utils.data import Dataset

class DatasetAbstract(abc.ABC):

    @abc.abstractmethod
    def get_train_dataset(self) -> Dataset:
        return None

    @abc.abstractmethod
    def get_val_dataset(self) -> Dataset:
        return None
