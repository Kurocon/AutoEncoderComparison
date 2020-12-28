import json
import logging
import os

import torch

from typing import Optional

from models.base_encoder import BaseEncoder


class BasicAutoEncoder(BaseEncoder):
    # Based on https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
    name = "BasicAutoEncoder"

    def __init__(self, name: Optional[str] = None, input_shape: int = 0):
        self.log = logging.getLogger(self.__class__.__name__)

        # Call superclass to initialize parameters.
        super(BasicAutoEncoder, self).__init__(name, input_shape)

        # Override parameters to custom values for this encoder type

        # TODO - Hoe kan ik het beste bepalen hoe groot de intermediate layers moeten zijn?
        # - Proportioneel van input grootte naar opgegeven bottleneck grootte?
        # - Uit een paper plukken
        # - Zelf kiezen (e.g. helft elke keer, fixed aantal layers)?
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
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Adam optimizer with learning rate 1e-3
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Mean Squared Error loss function
        # self.loss_function = torch.nn.MSELoss()

        self.after_init()

