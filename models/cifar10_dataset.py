import os

from typing import Optional

import numpy
import torchvision
from PIL import Image
from torchvision import transforms

from config import DATASET_STORAGE_BASE_PATH
from models.base_dataset import BaseDataset


class Cifar10Dataset(BaseDataset):

    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    #                                             torchvision.transforms.Normalize((0.5, ), (0.5, ))
    #                                             ])
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])

    def unpickle(self, filename):
        import pickle
        with open(filename, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def load(self, name: Optional[str] = None, path: Optional[str] = None):
        if name is not None:
            self.name = name
        if path is not None:
            self._source_path = path

        self._data = []
        for i in range(1, 6):
            data = self.unpickle(os.path.join(DATASET_STORAGE_BASE_PATH,
                                              self._source_path,
                                              f"data_batch_{i}"))
            self._data.extend(data[b'data'])

        self._trainset = self.__class__.get_new(name=f"{self.name} Training", data=self._data[:],
                                                source_path=self._source_path)

        test_data = self.unpickle(os.path.join(DATASET_STORAGE_BASE_PATH,
                                               self._source_path,
                                               f"test_batch"))
        self._data.extend(test_data[b'data'])
        self._testset = self.__class__.get_new(name=f"{self.name} Testing", data=test_data[b'data'][:],
                                               source_path=self._source_path)

        self.log.info(f"Loaded {self}, divided into {self._trainset} and {self._testset}")

    def __getitem__(self, item):
        # Get image data
        img = self._data[item]

        img_r, img_g, img_b = img.reshape((3, 1024))
        img_r = img_r.reshape((32, 32))
        img_g = img_g.reshape((32, 32))
        img_b = img_b.reshape((32, 32))

        # Reshape to 32x32x3 image
        img = numpy.stack((img_r, img_g, img_b), axis=2)

        # Run transforms
        if self.transform is not None:
            img = self.transform(img)

        # Reshape the 32x32x3 image to a 1x3072 array for the Linear layer
        img = img.view(-1, 32 * 32 * 3)

        return img