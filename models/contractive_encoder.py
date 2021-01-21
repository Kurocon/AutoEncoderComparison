import logging

import torch

from typing import Optional

from torch.autograd import Variable

from models.base_encoder import BaseEncoder


class ContractiveAutoEncoder(BaseEncoder):
    # Based on https://github.com/avijit9/Contractive_Autoencoder_in_Pytorch/blob/master/CAE_pytorch.py
    name = "ContractiveAutoEncoder"

    def __init__(self, name: Optional[str] = None, input_shape: int = 0, loss_function=None, regularizer_weight: float = 1e-4):
        self.log = logging.getLogger(self.__class__.__name__)

        # Call superclass to initialize parameters.
        super(ContractiveAutoEncoder, self).__init__(name, input_shape, loss_function)

        self.regularizer_weight = regularizer_weight

        # CAE needs intermediate output of the encoder stage, so split up the network into encoder/decoder
        self.network = None
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_shape, out_features=input_shape // 2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=input_shape // 2, out_features=input_shape // 4, bias=False),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_shape // 4, out_features=input_shape // 2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=input_shape // 2, out_features=input_shape, bias=False),
            torch.nn.ReLU()
        )

        # Adam optimizer with learning rate 1e-3 (parameters changed so we need to re-declare it)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Loss function is the same as defined in the base encoder.

    # Because network is split up now, and a CAE needs the intermediate representations,
    # we need to override the forward method and return the intermediate state as well.
    def forward(self, features):
        encoder_out = self.encoder(features)
        decoder_out = self.decoder(encoder_out)
        return encoder_out, decoder_out

    # Because the model returns the intermediate state as well as the output,
    # we need to override some output processing functions
    def process_outputs_for_loss_function(self, outputs):
        return outputs[1]

    def process_outputs_for_testing(self, outputs):
        return outputs[1]

    # Finally we define the loss process function, which adds the contractive loss.
    def process_loss(self, train_loss, features, outputs):
        """
        Evaluates the CAE loss, which is the summation of the MSE and the weighted L2-norm of the Jacobian of the
        hidden units with respect to the inputs.

        Reference: http://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder

        :param train_loss: The (MSE) loss as returned by the loss function of the model
        :param features: The input features
        :param outputs: The raw outputs as returned by the model (in this case includes the hidden encoder output)
        """
        hidden_output = outputs[0]
        # Weights of the second Linear layer in the encoder (index 2)
        weights = self.state_dict()['encoder.2.weight']

        # Hadamard product
        hidden_output = hidden_output.reshape(hidden_output.shape[0], hidden_output.shape[2])
        dh = hidden_output * (1 - hidden_output)

        # Sum through input dimension to improve efficiency (suggested in reference)
        w_sum = torch.sum(Variable(weights) ** 2, dim=1)

        # Unsqueeze to avoid issues with torch.mv
        w_sum = w_sum.unsqueeze(1)

        # Calculate contractive loss
        contractive_loss = torch.sum(torch.mm(dh ** 2, w_sum), 0)

        return train_loss + contractive_loss.mul_(self.regularizer_weight)
