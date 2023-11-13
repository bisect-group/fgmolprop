from typing import List, Tuple

import hydra
import torch.nn.init as init
from omegaconf import DictConfig
from torch import nn
from torchmetrics import Metric, MetricCollection

from src.data.components.global_dicts import ACTIVATION_FUNCTIONS, TASK_DICT


def make_encoder_decoder(
    input_dim: int,
    hidden_dims: List[int],
    bottleneck_dim: int,
    activation: str,
) -> Tuple[nn.modules.container.Sequential, nn.modules.container.Sequential]:
    """Function for creating encoder and decoder models.

    :param input_dim: Input dimension
    :param hidden_dims: List of hidden dimensions
    :param bottleneck_dim: Dimension of bottleneck layer
    :param activation: Activation function to use
    :raises ValueError: Activation function not supported
    :return: Encoder and decoder models
    """

    encoder_layers = nn.ModuleList()
    decoder_layers = nn.ModuleList()
    output_dim = input_dim
    dec_shape = bottleneck_dim

    try:
        act_fn = ACTIVATION_FUNCTIONS[activation]()
    except KeyError:
        raise ValueError("Activation function not supported")

    for enc_dim in hidden_dims:
        encoder_layers.extend([nn.Linear(input_dim, enc_dim), nn.LayerNorm(enc_dim), act_fn])
        input_dim = enc_dim

    encoder_layers.append(nn.Linear(input_dim, bottleneck_dim))

    dec_dims = list(reversed(hidden_dims))
    for dec_dim in dec_dims:
        decoder_layers.extend([nn.Linear(dec_shape, dec_dim), nn.LayerNorm(dec_dim), act_fn])
        dec_shape = dec_dim

    decoder_layers.append(nn.Linear(dec_shape, output_dim))

    return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)


def make_predictor(
    fcn_input_dim: int,
    output_dims: List[int],
    activation: str,
    num_tasks: int,
    dropout: float,
) -> nn.modules.container.Sequential:
    """Function for creating predictor model.

    :param fcn_input_dim: Input dimension
    :param output_dims: List of output dimensions
    :param activation: Activation function to use
    :param num_tasks: Number of tasks
    :param dropout: Dropout rate
    :raises ValueError: Activation function not supported
    :return: Predictor model
    """

    try:
        act_fn = ACTIVATION_FUNCTIONS[activation]()
    except KeyError:
        raise ValueError("Activation function not supported")

    dropout_layer = nn.Dropout(dropout)

    layers = nn.ModuleList()
    for output_dim in output_dims:
        layers.extend(
            [
                nn.Linear(fcn_input_dim, output_dim),
                nn.LayerNorm(output_dim),
                act_fn,
            ]
        )
        fcn_input_dim = output_dim

    layers.extend([dropout_layer, nn.Linear(fcn_input_dim, num_tasks)])

    return nn.Sequential(*layers)


def tie_decoder_weights(encoder: nn.Sequential, decoder: nn.Sequential) -> None:
    """Function for tying the weights of the encoder and decoder.

    :param encoder: Encoder model
    :param decoder: Decoder model
    """
    for i, encoder_layer in enumerate(reversed(encoder)):
        if isinstance(decoder[i], nn.Linear):
            decoder[i].weight = nn.Parameter(encoder_layer.weight.T)


def weight_init(model: nn.Module, activation: str) -> None:
    """Function for initializing the weights of the model.

    :param model: Model to initialize
    :param activation: Activation function to use
    """
    if activation == "selu":
        nonlinearity = "linear"
    else:
        nonlinearity = "leaky_relu"

    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nonlinearity)
            if m.bias is not None:
                init.constant_(m.bias, 0)


def load_metrics(metrics_cfg: DictConfig, dataset: str) -> Tuple[Metric, Metric, MetricCollection]:
    """Function for loading metrics.

    :param metrics_cfg: Metrics config
    :param dataset: Dataset to train on
    :raises RuntimeError: Requires valid_best metric that would track best state of Main Metric
    :return: Main metric, valid metric best, additional metrics
    """
    num_classes = TASK_DICT[dataset][0]
    task_type = TASK_DICT[dataset][1]

    main_metric = hydra.utils.instantiate(
        metrics_cfg.main,
        num_classes=num_classes,
        task=task_type,
        num_labels=num_classes,
    )
    if not metrics_cfg.get("valid_best"):
        raise RuntimeError(
            "Requires valid_best metric that would track best state of "
            "Main Metric. Usually it can be MaxMetric or MinMetric."
        )
    valid_metric_best = hydra.utils.instantiate(metrics_cfg.valid_best)

    additional_metrics = []
    if metrics_cfg.get("additional"):
        for _, metric_cfg in metrics_cfg.additional.items():
            additional_metrics.append(
                hydra.utils.instantiate(
                    metric_cfg,
                    num_classes=num_classes,
                    task=task_type,
                    num_labels=num_classes,
                )
            )

    return main_metric, valid_metric_best, MetricCollection(additional_metrics)
