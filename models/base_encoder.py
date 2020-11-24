from typing import Optional

from models.base_dataset import BaseDataset


class BaseEncoder:
    name = "BaseEncoder"

    def __init__(self, name: Optional[str] = None):
        if name is not None:
            self.name = name

    def __str__(self):
        return f"{self.name}"

    def train(self, dataset: BaseDataset):
        raise NotImplementedError()

    def test(self, dataset: BaseDataset):
        raise NotImplementedError()
