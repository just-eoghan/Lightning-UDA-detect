from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP
from bisect import bisect_right

class DaFrcnnDetectionModel(LightningModule):
    """
    LightningModule for domain adaptive object detection.
    """

    def __init__(
        self,
        da_net: torch.nn.Module,
        num_classes: int = 6,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        momentum: float = 0.9,
        trainable_backbone_layers: int = 0,
        lr_scheduler: bool = True
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = da_net
        self.val_map = MAP()
        self.test_map = MAP(class_metrics=True)

    def forward(self, x: torch.Tensor):
        self.model.eval()
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int):
        source_images, source_targets, _ = batch[0]
        target_images, target_targets, _ = batch[1]
        loss_dict = self.model(source_images + target_images, source_targets + target_targets)
        loss_dict = {"train/" + key: value for key, value in loss_dict.items()}
        self.log_dict(loss_dict, on_step=True, on_epoch=False, prog_bar=False)
        loss = sum(loss for loss in loss_dict.values())
        self.log("trian/loss_sum", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        images, targets, _ = batch
        preds = self.model(images)
        self.val_map.update(preds, targets)

    def on_validation_epoch_end(self):
        val_map = self.val_map.compute()
        val_dict = {"val/" + key: value for key, value in val_map.items()}
        self.log("val/map50", val_map["map_50"], on_step=False, on_epoch=True, prog_bar=False)
        self.log_dict(val_dict, on_step=False, on_epoch=True, prog_bar=False)
        self.val_map.reset()

    def test_step(self, batch: Any, batch_idx: int):
        images, targets, _ = batch
        preds = self.model(images)
        self.test_map.update(preds, targets)
        return {
            "test_images": images,
            "test_gt": targets,
            "test_outs": preds
        }

    def test_epoch_end(self, outputs: List[Any]):
        test_map = self.test_map.compute()
        self.log("test/map50", test_map["map_50"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/map", test_map, on_step=False, on_epoch=True, prog_bar=False)
        self.test_map.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        if not self.hparams.lr_scheduler:
            optimizer = torch.optim.SGD(
                params=self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay
            )
            return { "optimizer": optimizer }
        else:
            params = []
            for key, value in self.named_parameters():
                if not value.requires_grad:
                    continue
                lr = self.hparams.lr
                weight_decay = self.hparams.weight_decay
                if "bias" in key:
                    lr = self.hparams.lr * 2
                    weight_decay = 0
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

            optimizer = torch.optim.SGD(
                params=params, lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay
            )

            scheduler = WarmupMultiStepLR(optimizer, (30000, 40000))

            return {"optimizer": optimizer, "lr_scheduler": scheduler}

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]