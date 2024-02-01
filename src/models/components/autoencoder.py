from typing import Tuple

import torch
from torch import nn

from src.data.components.global_dicts import MFG_INPUT_DIM, TASK_DICT
from src.models.components.utils import (
    make_encoder_decoder,
    make_predictor,
    tie_decoder_weights,
)


class FGRModel(nn.Module):
    """An autoencoder model for FGR."""

    def __init__(
        self,
        fg_input_dim: int,
        num_feat_dim: int,
        method: str,
        tokenize_dataset: str,
        frequency: int,
        dataset: str,
        descriptors: bool,
        hidden_dim1: int,
        hidden_dim2: int,
        hidden_dim3: int,
        bottleneck_dim: int,
        output_dim1: int,
        output_dim2: int,
        output_dim3: int,
        dropout: float,
        activation: str,
        tie_weights: bool,
    ) -> None:
        """Initialize 'FGRModel'.

        :param fg_input_dim: FG input dimension
        :param num_feat_dim: Number of descriptor features
        :param method: Method for representation learning
        :param tokenize_dataset: Tokenization dataset
        :param frequency: Frequency for tokenization
        :param dataset: Dataset to train on
        :param descriptors: Whether to use descriptors
        :param hidden_dim1: Hidden dimension 1
        :param hidden_dim2: Hidden dimension 2
        :param hidden_dim3: Hidden dimension 3
        :param bottleneck_dim: Dimension of bottleneck layer
        :param output_dim1: Output dimension 1
        :param output_dim2: Output dimension 2
        :param output_dim3: Output dimension 3
        :param dropout: Dropout rate
        :param activation: Activation function to use
        :param tie_weights: Whether to tie weights of encoder and decoder
        :raises ValueError: Method not supported
        """
        super().__init__()

        self.method = method
        self.descriptors = descriptors
        self.tie_weights = tie_weights
        self.num_tasks, self.task_type, self.regression = TASK_DICT[dataset]
        hidden_dims = sorted([hidden_dim1, hidden_dim2, hidden_dim3], reverse=True)
        output_dims = sorted([output_dim1, output_dim2, output_dim3], reverse=True)

        if self.method == "FG":
            input_dim = fg_input_dim
        elif self.method == "MFG":
            input_dim = MFG_INPUT_DIM[tokenize_dataset][frequency]
        elif self.method == "FGR":
            input_dim = fg_input_dim + MFG_INPUT_DIM[tokenize_dataset][frequency]
        else:
            raise ValueError("Method not supported")
        self.encoder, self.decoder = make_encoder_decoder(
            input_dim, hidden_dims, bottleneck_dim, activation
        )

        # Tie the weights of the encoder and decoder
        if self.tie_weights:
            tie_decoder_weights(self.encoder, self.decoder)

        if self.descriptors:
            fcn_input_dim = bottleneck_dim + num_feat_dim
        else:
            fcn_input_dim = bottleneck_dim

        self.predictor = make_predictor(
            fcn_input_dim, output_dims, activation, self.num_tasks, dropout
        )

    def forward(
        self, x: torch.Tensor, num_feat: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform forward pass.

        :param x: Input tensor
        :param num_feat: Input Descriptors, defaults to None
        :return: Prediction, latent representation, reconstructed input
        """
        z_d = self.encoder(x)
        x_hat = self.decoder(z_d)
        if self.descriptors:
            assert num_feat is not None, "Descriptors not provided"
            z_d = torch.cat([z_d, num_feat], dim=1)  # Concatenate descriptors
        output = self.predictor(z_d)
        return output, z_d, x_hat


class FGRPretrainModel(nn.Module):
    """An autoencoder model for FGR pretraining."""

    def __init__(
        self,
        fg_input_dim: int,
        method: str,
        tokenize_dataset: str,
        frequency: int,
        hidden_dim1: int,
        hidden_dim2: int,
        hidden_dim3: int,
        bottleneck_dim: int,
        activation: str,
        tie_weights: bool,
    ) -> None:
        """Initialize 'FGRPretrainModel'.

        :param fg_input_dim: FG input dimension
        :param method: Method for representation learning
        :param tokenize_dataset: Tokenization dataset
        :param frequency: Frequency for tokenization
        :param hidden_dim1: Hidden dimension 1
        :param hidden_dim2: Hidden dimension 2
        :param hidden_dim3: Hidden dimension 3
        :param bottleneck_dim: Dimension of bottleneck layer
        :param activation: Activation function to use
        :param tie_weights: Whether to tie weights of encoder and decoder
        :raises ValueError: Method not supported
        """
        super().__init__()

        self.tie_weights = tie_weights
        hidden_dims = sorted([hidden_dim1, hidden_dim2, hidden_dim3], reverse=True)

        if method == "FG":
            self.input_dim = fg_input_dim
        elif method == "MFG":
            self.input_dim = MFG_INPUT_DIM[tokenize_dataset][frequency]
        elif method == "FGR":
            self.input_dim = fg_input_dim + MFG_INPUT_DIM[tokenize_dataset][frequency]
        else:
            raise ValueError("Method not supported")
        self.encoder, self.decoder = make_encoder_decoder(
            self.input_dim, hidden_dims, bottleneck_dim, activation
        )
        # Tie the weights of the encoder and decoder
        if self.tie_weights:
            tie_decoder_weights(self.encoder, self.decoder)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform forward pass.

        :param x: Input tensor
        :return: Latent representation, reconstructed input
        """
        z_d = self.encoder(x)
        x_hat = self.decoder(z_d)
        return z_d, x_hat
