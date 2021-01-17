import torch
from torch import Tensor

from models.base_corruption import BaseCorruption
from models.base_dataset import BaseDataset
import numpy


def add_noise(image):
    if isinstance(image, Tensor):
        image = image.numpy()
    image = image.astype(numpy.float32)
    mean, variance = 0, 0.1
    sigma = variance ** 0.5
    noise = numpy.random.normal(mean, sigma, image.shape).reshape(image.shape)
    return numpy.clip(image + noise, 0, 1)


class GaussianCorruption(BaseCorruption):
    """
    Corruption model that adds Gaussian noise to the dataset.
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
