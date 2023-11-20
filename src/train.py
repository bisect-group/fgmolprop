from copy import deepcopy
from typing import Any, Dict, List, Optional

import hydra
import lightning as L
import rootutils
import torch
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.models.components.utils import load_metrics_criterion  # noqa: E402
from src.utils import RankedLogger  # noqa: E402
from src.utils import extras  # noqa: E402
from src.utils import get_metric_value  # noqa: E402
from src.utils import instantiate_callbacks  # noqa: E402
from src.utils import instantiate_loggers  # noqa: E402
from src.utils import log_hyperparameters  # noqa: E402
from src.utils import task_wrapper  # noqa: E402; noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Dict[str, Any]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A Dict[str, Any] with the metric values obtained during training and testing.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating metrics...")
    (
        criterion,
        main_metric,
        valid_metric_best,
        additional_metrics,
    ) = load_metrics_criterion(cfg.metrics, cfg.data.dataset)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        criterion=criterion,
        main_metric=main_metric,
        valid_metric_best=valid_metric_best,
        additional_metrics=additional_metrics,
    )

    metric_dict: Dict[str, Any] = {}

    for fold_idx in range(cfg.get("n_folds")):
        log.info(f"Instantiating datamodule fold_{fold_idx} <{cfg.data._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, fold_idx=fold_idx)

        _model = deepcopy(model)

        log.info("Instantiating callbacks...")
        callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

        log.info("Instantiating loggers...")
        logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

        object_dict = {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": _model,
            "callbacks": callbacks,
            "logger": logger,
            "trainer": trainer,
        }

        if logger:
            log.info("Logging hyperparameters!")
            log_hyperparameters(object_dict)

        if cfg.get("train"):
            log.info("Starting training!")
            trainer.fit(model=_model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

        train_metrics = trainer.callback_metrics
        for key, value in train_metrics.items():
            if key in metric_dict:
                metric_dict[key] += value.float()
            else:
                metric_dict[key] = value.float()

        if cfg.get("test"):
            log.info("Starting testing!")
            ckpt_path = trainer.checkpoint_callback.best_model_path  # type: ignore
            if ckpt_path == "":
                log.warning("Best ckpt not found! Using current weights for testing...")
                ckpt_path = None
            trainer.test(model=_model, datamodule=datamodule, ckpt_path=ckpt_path)
            log.info(f"Best ckpt path: {ckpt_path}")
            test_metrics = trainer.callback_metrics
            for key, value in test_metrics.items():
                if key in metric_dict:
                    metric_dict[key] += value.float()
                else:
                    metric_dict[key] = value.float()

        wandb.finish()

    if cfg.get("n_folds") > 1:
        for key in metric_dict.keys():
            metric_dict[key] /= cfg.get("n_folds")
    return metric_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")  # Set high precision for matrix multiplication
    main()
