from typing import Any, Dict, Tuple

import torch
import torchmetrics
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger


class FGRLitModule(LightningModule):
    """FGR Lightning Module.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        base_optimizer: torch.optim.Optimizer,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        recon_loss: torch.nn.Module,
        criterion: torch.nn.Module,
        main_metric: torchmetrics.Metric,
        valid_metric_best: torchmetrics.Metric,
        additional_metrics: torchmetrics.MetricCollection,
        loss_weights: Dict[str, float],
        compile: bool,
    ) -> None:
        """Initialize a `FGRLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param recon_loss_fn: The reconstruction loss function to use for training.
        :param criterion: The loss function to use for training.
        :param main_metric: The main metric to use for training.
        :param valid_metric_best: The validation metric to use for training.
        :param additional_metrics: Additional metrics to use for training.
        :param loss_weights: The weights to use for the different losses.
        :param compile: Whether to compile the model.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "net",
                "recon_loss",
                "criterion",
                "main_metric",
                "valid_metric_best",
                "additional_metrics",
            ],
        )
        self.automatic_optimization = False

        self.net = net

        # loss functions
        self.criterion = criterion
        self.recon_loss = recon_loss
        self.loss_weights = loss_weights

        # metric objects for calculating and averaging accuracy across batches
        self.train_metric = main_metric.clone()
        self.train_add_metrics = additional_metrics.clone()
        self.val_metric = main_metric.clone()
        self.val_add_metrics = additional_metrics.clone()
        self.test_metric = main_metric.clone()
        self.test_add_metrics = additional_metrics.clone()

        # for tracking best so far validation accuracy
        self.val_best = valid_metric_best.clone()

    def forward(self, x: Any) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of features.
        :return: A tensor of logits.
        """
        return self.net(*x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_metric.reset()
        self.val_add_metrics.reset()
        self.val_best.reset()

        if isinstance(self.trainer.logger, WandbLogger):
            self.trainer.logger.watch(
                model=self.trainer.model,  # type: ignore
                log="all",
                log_freq=100,
            )

    def ubc_loss(self, z_d: Any) -> Any:
        """Calculate the UBC loss.

        :param z_d: Latent space representation of the input.
        :return: The UBC loss.
        """
        cov = torch.cov(z_d)
        off_diag = cov - torch.diag(torch.diag(cov))
        ubc_loss = torch.sum(off_diag**2)
        return ubc_loss

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of features and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        targets = batch[-1]
        logits, z_d, x_hat = self.forward(batch[:-1])
        loss = self.criterion(logits, targets)
        loss += self.loss_weights["recon_loss"] * self.recon_loss(x_hat, batch[0])
        loss += self.loss_weights["ubc_loss"] * self.ubc_loss(z_d)
        return loss, logits, targets

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of features and target
        :param batch_idx: The index of the current batch.
        """
        optimizer = self.optimizers()

        def closure():
            loss, _, _ = self.model_step(batch)
            self.manual_backward(loss)
            return loss

        # first forward-backward pass
        loss, logits, targets = self.model_step(batch)
        self.manual_backward(loss)
        optimizer.step(closure=closure)  # type: ignore
        optimizer.zero_grad()  # type: ignore

        scheduler = self.lr_schedulers()
        scheduler.step()  # type: ignore

        # cast targets to int if not regression
        if not self.trainer.datamodule.is_regression:  # type: ignore
            targets = targets.int()

        # update and log metrics
        self.train_metric.update(logits, targets)
        self.train_add_metrics.update(logits, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/main", self.train_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_add_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, logits, targets = self.model_step(batch)

        # cast targets to int if not regression
        if not self.trainer.datamodule.is_regression:  # type: ignore
            targets = targets.int()

        # update and log metrics
        self.val_metric.update(logits, targets)
        self.val_add_metrics.update(logits, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/main", self.val_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_add_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        main = self.val_metric.compute()  # get current val main
        self.val_best(main)  # update best so far val main
        # log `val_main_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/main_best", self.val_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, logits, targets = self.model_step(batch)

        # cast targets to int if not regression
        if not self.trainer.datamodule.is_regression:  # type: ignore
            targets = targets.int()

        # update and log metrics
        self.test_metric.update(logits, targets)
        self.test_add_metrics.update(logits, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/main", self.test_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_add_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams["compile"] and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams["optimizer"](
            params=self.trainer.model.parameters(), base_optimizer=self.hparams["base_optimizer"]  # type: ignore
        )
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams["scheduler"](
                optimizer=optimizer,
                total_steps=self.trainer.estimated_stepping_batches,
            )
            lr_scheduler = {"scheduler": scheduler}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return {"optimizer": optimizer}
