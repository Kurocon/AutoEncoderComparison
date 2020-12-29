import logging
import math

import torch

from typing import Optional

from torch.nn.modules.loss import _Loss

from models.base_encoder import BaseEncoder


class SparseL1AutoEncoder(BaseEncoder):
    # Based on https://debuggercafe.com/sparse-autoencoders-using-l1-regularization-with-pytorch/
    name = "SparseL1AutoEncoder"

    def __init__(self, name: Optional[str] = None, input_shape: int = 0, regularization_parameter: float = 0.001):
        self.log = logging.getLogger(self.__class__.__name__)

        # Call superclass to initialize parameters.
        super(SparseL1AutoEncoder, self).__init__(name, input_shape)

        # Override parameters to custom values for this encoder type

        # Sparse encoder has larger intermediary layers, so let's increase them 1.5 times in the first layer,
        # and 2 times in the second layer (compared to the original input shape
        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_shape, out_features=math.floor(input_shape * 1.5)),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=math.floor(input_shape * 1.5), out_features=input_shape * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=input_shape * 2, out_features=math.floor(input_shape * 1.5)),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=math.floor(input_shape * 1.5), out_features=input_shape),
            torch.nn.ReLU()
        )

        # Adam optimizer with learning rate 1e-3 (parameters changed so we need to re-declare it)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Loss function is the same as defined in the base encoder.

        # Regularization parameter (lambda) for the L1 sparse loss function
        self.regularization_parameter = regularization_parameter

    def get_sparse_loss(self, images):
        def get_sparse_loss_rec(loss, values, children):
            for child in children:
                if isinstance(child, torch.nn.Sequential):
                    loss, values = get_sparse_loss_rec(loss, values, [x for x in child])
                elif isinstance(child, torch.nn.ReLU):
                    values = child(values)
                    loss += torch.mean(torch.abs(values))
                elif isinstance(child, torch.nn.Linear):
                    values = child(values)
                else:
                    # Ignore unknown layers in sparse loss calculation
                    pass
            return loss, values

        loss, values = get_sparse_loss_rec(loss=0, values=images, children=list(self.children()))
        return loss


    def process_loss(self, train_loss, features, outputs) -> _Loss:
        l1_loss = self.get_sparse_loss(features)
        # Add sparsity penalty
        return train_loss + (self.regularization_parameter * l1_loss)
