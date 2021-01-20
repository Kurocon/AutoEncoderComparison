import random

from torch import Tensor

from models.base_corruption import BaseCorruption
from models.base_dataset import BaseDataset
import numpy


def add_noise(image):
    if isinstance(image, Tensor):
        image = image.numpy()
    image = image.astype(numpy.float32)

    # 90% chance to corrupt something
    if random.random() < 0.9:
        corrupt_index1, corrupt_index2 = random.sample(range(len(image)), 2)
        image[corrupt_index1] = 0
        # 10% chance to corrupt a second column
        if random.random() < 0.1:
            image[corrupt_index2] = 0

    return image


class RandomCorruption(BaseCorruption):
    """
    Corruption model that clears random fields of data.
    """
    name = "Gaussian"

    @classmethod
    def corrupt_image(cls, image: Tensor):
        return add_noise(image.numpy())

    @classmethod
    def corrupt_dataset(cls, dataset: BaseDataset) -> BaseDataset:
        data = [cls.corrupt_image(x) for x in dataset]
        # data = list(map(add_noise, dataset._data))
        train_set = cls.corrupt_dataset(dataset.get_train()) if dataset.has_train() else None
        test_set = cls.corrupt_dataset(dataset.get_test()) if dataset.has_test() else None
        return dataset.__class__.get_new(
            name=f"{dataset.name} Corrupted",
            data=data,
            labels=dataset._labels,
            source_path=dataset._source_path,
            train_set=train_set,
            test_set=test_set)
