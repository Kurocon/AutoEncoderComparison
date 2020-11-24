from models.base_corruption import BaseCorruption
from models.base_dataset import BaseDataset
from models.base_encoder import BaseEncoder


class TestRun:
    dataset: BaseDataset = None
    encoder: BaseEncoder = None
    corruption: BaseCorruption = None

    def __init__(self, dataset: BaseDataset, encoder: BaseEncoder, corruption: BaseCorruption):
        self.dataset = dataset
        self.encoder = encoder
        self.corruption = corruption

    def run(self):
        if self.dataset is None:
            raise ValueError("Cannot run test! Dataset is not specified.")
        if self.encoder is None:
            raise ValueError("Cannot run test! AutoEncoder is not specified.")
        if self.corruption is None:
            raise ValueError("Cannot run test! Corruption method is not specified.")
        return self._run()

    def _run(self):
        raise NotImplementedError()

    def get_metrics(self):
        raise NotImplementedError()
