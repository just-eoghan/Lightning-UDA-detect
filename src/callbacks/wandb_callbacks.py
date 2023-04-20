import subprocess
from pathlib import Path
from typing import List


import numpy as np
import seaborn as sn
import torch
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import Logger, WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from wandb.sdk.data_types._dtypes import AnyType

def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, Logger):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq, log_graph=True)


class UploadCodeAsArtifact(Callback):
    """Upload all code files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str, use_git: bool = True):
        """

        Args:
            code_dir: the code directory
            use_git: if using git, then upload all files that are not ignored by git.
            if not using git, then upload all '*.py' file
        """
        self.code_dir = code_dir
        self.use_git = use_git

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")

        if self.use_git:
            # get .git folder path
            git_dir_path = Path(
                subprocess.check_output(["git", "rev-parse", "--git-dir"]).strip().decode("utf8")
            ).resolve()

            for path in Path(self.code_dir).resolve().rglob("*"):

                # don't upload files ignored by git
                # https://alexwlchan.net/2020/11/a-python-function-to-ignore-a-path-with-git-info-exclude/
                command = ["git", "check-ignore", "-q", str(path)]
                not_ignored = subprocess.run(command).returncode == 1

                # don't upload files from .git folder
                not_git = not str(path).startswith(str(git_dir_path))

                if path.is_file() and not_git and not_ignored:
                    code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        else:
            for path in Path(self.code_dir).resolve().rglob("*.py"):
                code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        experiment.log_artifact(code)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)


class LogBoundingBoxes(Callback):
    """Generate images with bounding box predictions overlayed"""

    def __init__(self):
        self.pred_boxes = []
        self.pred_labels = []
        self.scores = []
        self.images = []
        self.gt_boxes = []
        self.gt_labels = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Gather data from single batch."""

        if self.ready:
            if outputs is None:
                return
            for item in outputs["test_outs"]:
                self.pred_boxes.append(item["boxes"])
                self.pred_labels.append(item["labels"])
                self.scores.append(item["scores"])
            for image in outputs["test_images"]:
                self.images.append(image)
            for item in outputs["test_gt"]:
                self.gt_boxes.append(item["boxes"])
                self.gt_labels.append(item["labels"])
        return

    def on_test_epoch_end(self, trainer, pl_module):
        """Log bounding boxes."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment

            class_dict = {
                "empty": 0,
                "person": 1
            }

            image_classes = [{"id": int(v), "name": k} for k, v in class_dict.items()]
            class_id_to_label = {int(v): k for k, v in class_dict.items()}

            img_idx = 0
            bounding_data = []

            for image, v_boxes, v_labels, scores in zip(
                self.images, self.pred_boxes, self.pred_labels, self.scores,
            ):
                all_vboxes = []
                vboxes_50 = []
                vboxes_75 = []
                for b_i, box in enumerate(v_boxes):
                    # get coordinates and labels
                    box_data = {
                        "position": {
                            "minX": box[0].item(),
                            "maxX": box[2].item(),
                            "minY": box[1].item(),
                            "maxY": box[3].item(),
                        },
                        "class_id": v_labels[b_i].item(),
                        "domain": "pixel",
                        "scores": {"score": scores[b_i].item()},
                    }
                    all_vboxes.append(box_data)
                    if scores[b_i].item() >= 0.5:
                        vboxes_50.append(box_data)
                    if scores[b_i].item() >= 0.75:
                        vboxes_75.append(box_data)

                # there's prob a better way of doing this, makes it (120,160,3)
                np_image = np.swapaxes(np.swapaxes(image.cpu().numpy(), 0, -1), 0, 1)

                if len(all_vboxes) > 0:
                    box_image = wandb.Image(
                        np_image,
                        boxes={
                            "predictions": {
                                "box_data": all_vboxes,
                                "class_labels": class_id_to_label,
                            }
                        },
                        classes=image_classes,
                    )
                else:
                    box_image = wandb.Image(np_image)

                if len(vboxes_50) > 0:
                    box50_image = wandb.Image(
                        np_image,
                        boxes={
                            "predictions": {
                                "box_data": vboxes_50,
                                "class_labels": class_id_to_label,
                            }
                        },
                        classes=image_classes,
                    )
                else:
                    box50_image = wandb.Image(np_image)

                if len(vboxes_75) > 0:
                    box75_image = wandb.Image(
                        np_image,
                        boxes={
                            "predictions": {
                                "box_data": vboxes_75,
                                "class_labels": class_id_to_label,
                            }
                        },
                        classes=image_classes,
                    )
                else:
                    box75_image = wandb.Image(np_image)

                bounding_data.append([box_image, box50_image, box75_image])
                img_idx = img_idx + 1

            img_idx = 0  # reset image index
            for image, gt_boxes, gt_labels in zip(self.images, self.gt_boxes, self.gt_labels):
                all_gtboxes = []
                for b_i, box in enumerate(gt_boxes):
                    # get coordinates and labels
                    box_data = {
                        "position": {
                            "minX": box[0].item(),
                            "maxX": box[2].item(),
                            "minY": box[1].item(),
                            "maxY": box[3].item(),
                        },
                        "class_id": gt_labels[b_i].item(),
                        "domain": "pixel",
                    }
                    all_gtboxes.append(box_data)
                np_image = np.swapaxes(np.swapaxes(image.cpu().numpy(), 0, -1), 0, 1)

                gtbox_image = wandb.Image(
                    np_image,
                    boxes={
                        "ground_truth": {
                            "box_data": all_gtboxes,
                            "class_labels": class_id_to_label,
                        }
                    },
                    classes=image_classes,
                )

                bounding_data[img_idx].append(gtbox_image)
                img_idx = img_idx + 1

            columns = ["bbox_preds", "bbox_50", "bbox_75", "bbox_gt"]
            image_pred_table = wandb.Table(data=bounding_data, columns=columns, dtype=AnyType)

            experiment.log({f"test_set_bbox_preds/{experiment.name}": image_pred_table})
