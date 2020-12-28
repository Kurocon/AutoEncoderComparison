import json
import logging
import os

import torch

from typing import Optional

from torchvision.utils import save_image

from config import TRAIN_TEMP_DATA_BASE_PATH, TEST_TEMP_DATA_BASE_PATH, MODEL_STORAGE_BASE_PATH
from models.base_dataset import BaseDataset


class BaseEncoder(torch.nn.Module):
    # Based on https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
    name = "BaseEncoder"

    def __init__(self, name: Optional[str] = None, input_shape: int = 0):
        super(BaseEncoder, self).__init__()
        self.log = logging.getLogger(self.__class__.__name__)

        if name is not None:
            self.name = name

        assert input_shape != 0, f"Encoder {self.__class__.__name__} input_shape parameter should not be 0"

        self.input_shape = input_shape

        # Default fallbacks (can be overridden by sub implementations)

        # 4 layer NN, halving the input each layer
        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_shape, out_features=input_shape // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=input_shape // 2, out_features=input_shape // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=input_shape // 4, out_features=input_shape // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=input_shape // 2, out_features=input_shape)
        )

        # Use GPU acceleration if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Adam optimizer with learning rate 1e-3
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Mean Squared Error loss function
        self.loss_function = torch.nn.MSELoss()

    def after_init(self):
        self.log.info(f"Auto-encoder {self.__class__.__name__} initialized with {len(list(self.network.children()))} layers on {self.device.type}. Optimizer: {self.optimizer}, Loss function: {self.loss_function}")

    def forward(self, features):
        return self.network(features)

    def save_model(self, filename):
        torch.save(self.state_dict(), os.path.join(MODEL_STORAGE_BASE_PATH, f"{filename}.model"))
        with open(os.path.join(MODEL_STORAGE_BASE_PATH, f"{filename}.meta"), 'w') as f:
            f.write(json.dumps({
                'name': self.name,
                'input_shape': self.input_shape
            }))

    def load_model(self, filename=None):
        if filename is None:
            filename = f"{self.name}"
        try:
            loaded_model = torch.load(os.path.join(MODEL_STORAGE_BASE_PATH, f"{filename}.model"), map_location=self.device)
            self.load_state_dict(loaded_model)
            self.to(self.device)
            return True
        except OSError as e:
            self.log.error(f"Could not load model '{filename}': {e}")
            return False

    @classmethod
    def create_model_from_file(cls, filename, device=None):
        try:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            with open(os.path.join(MODEL_STORAGE_BASE_PATH, f"{filename}.meta")) as f:
                model_kwargs = json.loads(f.read())
            model = cls(**model_kwargs)
            loaded_model = torch.load(os.path.join(MODEL_STORAGE_BASE_PATH, f"{filename}.model"), map_location=device)
            model.load_state_dict(loaded_model)
            model.to(device)
            return model
        except OSError as e:
            log = logging.getLogger(cls.__name__)
            log.error(f"Could not load model '{filename}': {e}")
            return None

    def __str__(self):
        return f"{self.name}"

    def train_encoder(self, dataset: BaseDataset, epochs: int = 20, batch_size: int = 128, num_workers: int = 4):
        self.log.debug("Getting training dataset DataLoader.")
        train_loader = dataset.get_train_loader(batch_size=batch_size, num_workers=num_workers)

        # Puts module in training mode.
        self.log.debug("Putting model into training mode.")
        self.train()
        self.to(self.device, non_blocking=True)
        self.loss_function.to(self.device, non_blocking=True)

        losses = []

        outputs = None
        for epoch in range(epochs):
            self.log.debug(f"Start training epoch {epoch}...")
            loss = 0
            for batch_features in train_loader:
                # load batch features to the active device
                batch_features = batch_features.to(self.device)

                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                self.optimizer.zero_grad()

                # compute reconstructions
                outputs = self(batch_features)

                # compute training reconstruction loss
                train_loss = self.loss_function(outputs, batch_features)

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                self.optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()

            # compute the epoch training loss
            loss = loss / len(train_loader)

            # display the epoch training loss
            self.log.info("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
            losses.append(loss)

            # Every 5 epochs, save a test image
            if epoch % 5 == 0:
                img = outputs.cpu().data
                img = img.view(img.size(0), 3, 32, 32)
                save_image(img, os.path.join(TRAIN_TEMP_DATA_BASE_PATH,
                                             f'{self.name}_{dataset.name}_linear_ae_image{epoch}.png'))

        return losses


    def test_encoder(self, dataset: BaseDataset, batch_size: int = 128, num_workers: int = 4):
        self.log.debug("Getting testing dataset DataLoader.")
        test_loader = dataset.get_test_loader(batch_size=batch_size, num_workers=num_workers)

        self.log.debug(f"Start testing...")
        i = 0
        for batch in test_loader:
            img = batch.view(batch.size(0), 3, 32, 32)
            save_image(img, os.path.join(TEST_TEMP_DATA_BASE_PATH,
                                         f'{self.name}_{dataset.name}_test_input_{i}.png'))
            # load batch features to the active device
            batch = batch.to(self.device)
            outputs = self(batch)
            img = outputs.cpu().data
            img = img.view(outputs.size(0), 3, 32, 32)
            save_image(img, os.path.join(TEST_TEMP_DATA_BASE_PATH,
                                         f'{self.name}_{dataset.name}_test_reconstruction_{i}.png'))
            i += 1
            break
