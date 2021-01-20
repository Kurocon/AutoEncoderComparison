import os

from typing import Optional

import numpy
from pytorch_msssim import ssim
from torchvision import transforms
from torchvision.utils import save_image

from config import DATASET_STORAGE_BASE_PATH
from models.base_dataset import BaseDataset


class Cifar10Dataset(BaseDataset):
    name = "CIFAR-10"

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
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

    def get_input_shape(self):
        return 3072  # 32x32x3 (32x32px, 3 colors)

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

    def get_as_image_array(self, item):
        # Get image data
        img = self._data[item]

        img = img.reshape((3, 1024))

        # Run transforms
        if self.transform is not None:
            img = self.transform(img)

        # Reshape the 32x32x3 image to a 1x3072 array for the Linear layer
        img = img.view(-1, 3, 32, 32)

        return img

    def save_batch_to_sample(self, batch, filename):
        img = batch.view(batch.size(0), 3, 32, 32)[:48]
        save_image(img, f"{filename}.png")

    def calculate_score(self, originals, reconstruction, device):
        # Calculate SSIM
        originals = originals.view(originals.size(0), 3, 32, 32).to(device)
        reconstruction = reconstruction.view(reconstruction.size(0), 3, 32, 32).to(device)
        batch_average_score = ssim(originals, reconstruction, data_range=1, size_average=True)
        return batch_average_score
