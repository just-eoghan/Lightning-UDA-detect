from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP
from bisect import bisect_right

from src.models.modules.da_mic.masking import Masking
from src.models.modules.da_mic.teacher import EMATeacher

def process_pred2label(target_output, threshold=0.7):
    masks = []
    pseudo_labels_list = []
    for idx, bbox_l in enumerate(target_output):
        pred_bboxes = bbox_l['boxes']
        labels = bbox_l['labels']
        scores = bbox_l['scores']
        filtered_idx = scores>=threshold
        filtered_bboxes = pred_bboxes[filtered_idx]
        filtered_labels = labels[filtered_idx]
        domain_labels = torch.ones_like(filtered_labels, dtype=torch.uint8).to(filtered_labels.device)
        new_bbox_list = {
            'boxes': filtered_bboxes,
            'labels': filtered_labels,
            'is_source': domain_labels
        }
        if len(new_bbox_list["boxes"])>0:
            pseudo_labels_list.append(new_bbox_list)
            masks.append(idx)
    return pseudo_labels_list, masks

class MicDaFrcnnDetectionModel(LightningModule):
    """
    LightningModule for domain adaptive object detection.
    """

    def __init__(
        self,
        da_net: torch.nn.Module,
        num_classes: int = 9,
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

        self.masking = Masking(
            block_size=32,
            ratio=0.5,
            color_jitter_s=0.2,
            color_jitter_p=0.2,
            blur=True,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        self.teacher = EMATeacher(self.model, alpha=0.9)

    def forward(self, x: torch.Tensor):
        self.model.eval()
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int):

        self.teacher.eval()

        source_images, source_targets, _ = batch[0]
        target_images, target_targets, _ = batch[1]

        masked_target_images = self.masking(torch.stack(target_images))
        self.teacher.update_weights(self.model, self.global_step)
        target_output = self.teacher(target_images)
        target_pseudo_labels, pseudo_masks = process_pred2label(target_output, threshold=0.8)

        self.model.train()
        record_dict = self.model(source_images + target_images, source_targets + target_targets)

        if len(target_pseudo_labels)>0:
            masked_images = masked_target_images[pseudo_masks]
            masked_taget = target_pseudo_labels
            masked_loss_dict = self.model(masked_images, masked_taget, use_pseudo_labeling_weight='prob', with_DA_ON=False)

            new_record_all_unlabel_data = {}
            for key in masked_loss_dict.keys():
                new_record_all_unlabel_data[key + "_mask"] = masked_loss_dict[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)
        
        loss_dict = {}
        for key in record_dict.keys():
            if key.startswith("loss"):
                if key == "loss_box_reg_mask" or key == "loss_rpn_box_reg_mask":
                    # pseudo bbox regression <- 0
                    loss_dict[key] = record_dict[key] * 0
                elif key.endswith('_mask') and 'da' in key:
                    loss_dict[key] = record_dict[key] * 0
                elif key == 'loss_classifier_mask' or key == 'loss_objectness_mask':
                    loss_dict[key] = record_dict[key] * 1
                else:  # supervised loss
                    loss_dict[key] = record_dict[key] * 1

        loss = sum(loss for loss in loss_dict.values())

        if 'loss_classifier_mask' in loss_dict:
            self.log("train/loss_classifier_mask", loss_dict['loss_classifier_mask'], on_step=True, on_epoch=False, prog_bar=False)
        if 'loss_box_reg_mask' in loss_dict:
            self.log("train/loss_box_reg_mask", loss_dict['loss_box_reg_mask'], on_step=True, on_epoch=False, prog_bar=False)
        if 'loss_objectness_mask' in loss_dict:
            self.log("train/loss_objectness_mask", loss_dict['loss_objectness_mask'], on_step=True, on_epoch=False, prog_bar=False)
        if 'loss_rpn_box_reg_mask' in loss_dict:
            self.log("train/loss_rpn_box_reg_mask", loss_dict['loss_rpn_box_reg_mask'], on_step=True, on_epoch=False, prog_bar=False)

        log_dict = {}
        for key in loss_dict:
            if key not in ['loss_classifier_mask', 'loss_box_reg_mask', 'loss_objectness_mask', 'loss_rpn_box_reg_mask']:
                log_dict[key] = loss_dict[key]

        log_dict = {"train/" + key: value for key, value in log_dict.items()}
        self.log_dict(log_dict, on_step=True, on_epoch=False, prog_bar=False)
        self.log("trian/loss_sum", loss, on_step=True, on_epoch=False, prog_bar=False)
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

    def on_test_epoch_end(self, outputs: List[Any]):
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