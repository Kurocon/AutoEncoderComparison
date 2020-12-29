import logging

from typing import Optional

from models.base_encoder import BaseEncoder


class BasicAutoEncoder(BaseEncoder):
    # Based on https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
    name = "BasicAutoEncoder"

    def __init__(self, name: Optional[str] = None, input_shape: int = 0):
        self.log = logging.getLogger(self.__class__.__name__)

        # Call superclass to initialize parameters.
        super(BasicAutoEncoder, self).__init__(name, input_shape)

        # Network, optimizer and loss function are the same as defined in the base encoder.
