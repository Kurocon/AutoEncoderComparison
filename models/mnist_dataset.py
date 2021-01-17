import os

from typing import Optional

from pytorch_msssim import ssim
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from config import DATASET_STORAGE_BASE_PATH
from models.base_dataset import BaseDataset


class MNISTDataset(BaseDataset):
    name = "MNIST"

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    def load(self, name: Optional[str] = None, path: Optional[str] = None):
        if name is not None:
            self.name = name
        if path is not None:
            self._source_path = path

        train_dataset = MNIST(root=os.path.join(DATASET_STORAGE_BASE_PATH, self._source_path), train=True, download=True)
        train_data = [x for x in train_dataset.data]
        self._data = train_data

        self._trainset = self.__class__.get_new(name=f"{self.name} Training", data=train_data[:],
                                                source_path=self._source_path)

        test_dataset = MNIST(root=os.path.join(DATASET_STORAGE_BASE_PATH, self._source_path), train=False, download=True)
        test_data = [x for x in test_dataset.data]
        self._data.extend(test_data)

        self._testset = self.__class__.get_new(name=f"{self.name} Testing", data=test_data[:],
                                               source_path=self._source_path)

        self.log.info(f"Loaded {self}, divided into {self._trainset} and {self._testset}")

    def __getitem__(self, item):
        # Get image data
        img = self._data[item]

        # Run transforms
        if self.transform is not None:
            img = self.transform(img)

        # Reshape the 28x28x1 image to a 1x784 array for the Linear layer
        img = img.view(-1, 28 * 28)

        return img

    def save_batch_to_sample(self, batch, filename):
        img = batch.view(batch.size(0), 1, 28, 28)
        save_image(img, f"{filename}.png")

    def calculate_score(self, originals, reconstruction, device):
        # Calculate SSIM
        originals = originals.view(originals.size(0), 1, 28, 28).to(device)
        reconstruction = reconstruction.view(reconstruction.size(0), 1, 28, 28).to(device)
        batch_average_score = ssim(originals, reconstruction, data_range=1, size_average=True)
        return batch_average_score
