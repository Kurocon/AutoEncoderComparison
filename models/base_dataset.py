import math
from typing import Union, Optional


class BaseDataset:

    # Train amount is either a proportion of data that should be used as training data (between 0 and 1),
    # or an integer indicating how many entries should be used as training data (e.g. 1000, 2000)
    #
    # So 0.2 would mean 20% of all data in the dataset (200 if dataset is 1000 entries) is used as training data,
    # and 1000 would mean that 1000 entries are used as training data, regardless of the size of the dataset.
    TRAIN_AMOUNT = 0.2

    name = "BaseDataset"
    _source_path = None
    _data = None
    _trainset: 'BaseDataset' = None
    _testset: 'BaseDataset' = None

    def __init__(self, name: Optional[str] = None):
        if name is not None:
            self.name = name

    def __str__(self):
        if self._data is not None:
            return f"{self.name} ({len(self._data)} objects)"
        else:
            return f"{self.name} (no data loaded)"

    @classmethod
    def get_new(cls, name: str, data: Optional[list] = None, source_path: Optional[str] = None,
                train_set: Optional['BaseDataset'] = None, test_set: Optional['BaseDataset'] = None):
        dset = cls()
        dset._data = data
        dset._source_path = source_path
        dset._trainset = train_set
        dset._testset = test_set
        return dset

    def load(self, name: str, path: str):
        self.name = str
        self._source_path = path
        raise NotImplementedError()

    def _subdivide(self, amount: Union[int, float]):
        if self._data is None:
            raise ValueError("Cannot subdivide! Data not loaded, call `load()` first to load data")

        if isinstance(amount, float) and 0 < amount < 1:
            size_train = math.floor(len(self._data) * amount)
            train_data = self._data[:size_train]
            test_data = self._data[size_train:]
        elif isinstance(amount, int) and amount > 0:
            train_data = self._data[:amount]
            test_data = self._data[amount:]
        else:
            raise ValueError("Cannot subdivide! Invalid amount given, "
                             "must be either a fraction between 0 and 1, or an integer.")

        self._trainset = self.__class__.get_new(name=f"{self.name} Training", data=train_data, source_path=self._source_path)
        self._testset = self.__class__.get_new(name=f"{self.name} Testing", data=test_data, source_path=self._source_path)

    def get_train(self) -> 'BaseDataset':
        if not self._trainset or not self._testset:
            self._subdivide(self.TRAIN_AMOUNT)
        return self._trainset

    def get_test(self) -> 'BaseDataset':
        if not self._trainset or not self._testset:
            self._subdivide(self.TRAIN_AMOUNT)
        return self._testset
