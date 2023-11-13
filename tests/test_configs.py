import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.models.components.utils import load_metrics_criterion


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    (
        criterion,
        main_metric,
        valid_metric_best,
        additional_metrics,
    ) = load_metrics_criterion(cfg_train.metrics, cfg_train.data.dataset)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(
        cfg_train.model,
        criterion=criterion,
        main_metric=main_metric,
        valid_metric_best=valid_metric_best,
        additional_metrics=additional_metrics,
    )
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest fixture.

    :param cfg_train: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    (
        criterion,
        main_metric,
        valid_metric_best,
        additional_metrics,
    ) = load_metrics_criterion(cfg_eval.metrics, cfg_eval.data.dataset)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(
        cfg_eval.model,
        criterion=criterion,
        main_metric=main_metric,
        valid_metric_best=valid_metric_best,
        additional_metrics=additional_metrics,
    )
    hydra.utils.instantiate(cfg_eval.trainer)
