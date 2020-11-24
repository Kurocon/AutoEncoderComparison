from models.base_dataset import BaseDataset


class BaseCorruption:
    """
    Base corruption model that is not implemented.
    """
    name = "BaseCorruption"

    def __init__(self, name: str = None):
        if name is not None:
            self.name = name

    def __str__(self):
        return f"{self.name}"

    @classmethod
    def corrupt(cls, dataset: BaseDataset) -> BaseDataset:
        raise NotImplementedError()


class NoCorruption(BaseCorruption):
    """
    Corruption model that does not corrupt the dataset at all.
    """
    name = "No corruption"

    @classmethod
    def corrupt(cls, dataset: BaseDataset) -> BaseDataset:
        return dataset
