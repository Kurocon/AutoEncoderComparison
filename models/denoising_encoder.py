import logging

import torch

from typing import Optional

from torch import Tensor

from main import load_dotted_path
from models.base_corruption import BaseCorruption, NoCorruption
from models.base_encoder import BaseEncoder


class DenoisingAutoEncoder(BaseEncoder):
    # Based on https://github.com/pranjaldatta/Denoising-Autoencoder-in-Pytorch/blob/master/DenoisingAutoencoder.ipynb
    name = "DenoisingAutoEncoder"

    def __init__(self, name: Optional[str] = None, input_shape: int = 0,
                 input_corruption_model: BaseCorruption = NoCorruption):
        self.log = logging.getLogger(self.__class__.__name__)

        # Call superclass to initialize parameters.
        super(DenoisingAutoEncoder, self).__init__(name, input_shape)

        # Network, optimizer and loss function are the same as defined in the base encoder.

        # Corruption used for data corruption during training
        if isinstance(input_corruption_model, str):
            self.input_corruption_model = load_dotted_path(input_corruption_model)
        else:
            self.input_corruption_model = input_corruption_model

    # Need to corrupt features used for training, to add 'noise' to training data (comparison features are unchanged)
    def process_train_features(self, features: Tensor) -> Tensor:
        return torch.tensor([self.input_corruption_model.corrupt_image(x) for x in features], dtype=torch.float32)
