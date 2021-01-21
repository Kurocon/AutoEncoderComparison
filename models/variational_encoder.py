import logging

import torch

from typing import Optional

from models.base_encoder import BaseEncoder


class VariationalAutoEncoder(BaseEncoder):
    # Based on https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
    # and https://github.com/pytorch/examples/blob/master/vae/main.py
    name = "VariationalAutoEncoder"

    def __init__(self, name: Optional[str] = None, input_shape: int = 0, loss_function=None):
        self.log = logging.getLogger(self.__class__.__name__)

        # Call superclass to initialize parameters.
        super(VariationalAutoEncoder, self).__init__(name, input_shape, loss_function)

        # VAE needs intermediate output of the encoder stage, so split up the network into encoder/decoder
        # with no ReLU layer at the end of the encoder so we have access to the mu and variance.
        # We also split the last layer of the encoder in two, so we can make two passes.
        # One to determine the mu and one to determine the variance
        self.network = None
        self.encoder1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_shape, out_features=input_shape // 2, bias=False),
            torch.nn.ReLU()
        )
        self.encoder2_1 = torch.nn.Linear(in_features=input_shape // 2, out_features=input_shape // 4, bias=False)
        self.encoder2_2 = torch.nn.Linear(in_features=input_shape // 2, out_features=input_shape // 4, bias=False)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_shape // 4, out_features=input_shape // 2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=input_shape // 2, out_features=input_shape, bias=False),
            torch.nn.ReLU()
        )

        # Adam optimizer with learning rate 1e-3 (parameters changed so we need to re-declare it)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Loss function is the same as defined in the base encoder.

    # Reparameterize takes a mu and variance, and returns a sample from the distribution N(mu(X), log_var(x))
    def reparameterize(self, mu, log_var):
        # z = μ(X) + Σ1/2(X)∗e
        std_dev = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std_dev)
        return mu + (std_dev * epsilon)

    # Because network is split up, and a VAE needs the intermediate representations, needs to modify the input to the
    # decoder, and needs to return the parameters used for the sample, we need to override the forward method.
    def forward(self, features):
        encoder_out = self.encoder1(features)

        # Use the two last layers of the encoder to determine the mu and log_var
        mu = self.encoder2_1(encoder_out)
        log_var = self.encoder2_2(encoder_out)

        # Get a sample from the distribution with mu and log_var, for use in the decoder
        sample = self.reparameterize(mu, log_var)

        decoder_out = self.decoder(sample)
        return decoder_out, mu, log_var

    # Because the model returns the mu and log_var in addition to the output,
    # we need to override some output processing functions
    def process_outputs_for_loss_function(self, outputs):
        return outputs[0]

    def process_outputs_for_testing(self, outputs):
        return outputs[0]

    # After the loss function is executed, we modify the loss with KL divergence
    def process_loss(self, train_loss, features, outputs):
        # Loss = Reconstruction loss + KL divergence loss summed over all elements and batch
        _, mu, log_var = outputs

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return train_loss + kl_divergence
