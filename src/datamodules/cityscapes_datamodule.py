import os
from typing import Optional

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datamodules.datasets.da_frcnn_dataset import DaFrcnnDataset

class Collater:
    # https://shoarora.github.io/2020/02/01/collate_fn.html
    def __call__(self, batch):
        return tuple(zip(*batch))


class DaCityscapesDatamodule(LightningDataModule):
    """
    DaFrcnnDatamodule for domain adaptive faster rcnn object detection.

    """

    def __init__(
        self,
        source_data_dir: str = "data/",
        target_data_dir: str = "data/",
        train_batch_size: int = 2,
        val_batch_size: int = 2,
        test_batch_size: int = 2,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_classes: int = 2,
        image_width: int = 160,
        image_height: int = 120
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.transforms = A.Compose(
            [
                A.Resize(800,1600),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="coco", label_fields=["category_ids"]
            ),
        )

        self.no_transforms = A.Compose(
            [
                A.Resize(800,1600),
                A.Normalize(),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="coco", label_fields=["category_ids"]
            ),
        )

        self.collater = Collater()

        # self.dims is returned when you call datamodule.size()
        self.dims = (3, self.hparams.image_height, self.hparams.image_width)
        self.data_train_source: Optional[Dataset] = None
        self.data_train_target: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return self.hparams.num_classes

    def setup(self, stage: Optional[str] = None):
        if not self.data_train_source and not self.data_train_target and not self.data_val:
            self.data_train_source = DaFrcnnDataset(
                self.hparams.source_data_dir + "images/",
                self.hparams.source_data_dir + "/annotations/instancesonly_filtered_gtFine_train.json",
                remove_images_without_annotations=True,
                is_source=True,
                transform=self.transforms
            )

            self.data_train_target = DaFrcnnDataset(
                self.hparams.target_data_dir + "images/",
                self.hparams.target_data_dir + "/annotations/instancesonly_filtered_gtFine_train.json",
                remove_images_without_annotations=True,
                is_source=False,
                transform=self.transforms
            )

            self.data_val = DaFrcnnDataset(
                self.hparams.target_data_dir + "images/",
                self.hparams.target_data_dir + "/annotations/instancesonly_filtered_gtFine_val.json",
                remove_images_without_annotations=True,
                is_source=False,
                transform=self.no_transforms
            )

            return

    def train_dataloader(self):
        return [
            DataLoader(
                dataset=self.data_train_source,
                batch_size=int(self.hparams.train_batch_size / 2),
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
                # https://github.com/pytorch/vision/issues/2624#issuecomment-681811444
                collate_fn=self.collater,
            ),
            DataLoader(
                dataset=self.data_train_target,
                batch_size=int(self.hparams.train_batch_size / 2),
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
                # https://github.com/pytorch/vision/issues/2624#issuecomment-681811444
                collate_fn=self.collater,
            )
        ]

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collater,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collater,
        )