import os
import os.path
from typing import Any, Callable, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.ops import box_convert

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    return False

class DaFrcnnDataset(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        remove_images_without_annotations: bool = True,
        is_source: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        persons_only: Optional[bool] = False,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.persons_only = persons_only
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms
        self.is_source = is_source

    def _load_image(self, id: int) -> Tuple[Image.Image, str]:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return (Image.open(os.path.join(self.root, path)).convert("RGB"), path)

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def _get_category_name(self, id: int) -> str:
            return self.class_dict.get(id)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        id = self.ids[index]
        image, file_name = self._load_image(id)
        targets = self._load_target(id)

        bboxes = []
        category_ids = []

        for target in targets:
            category_ids.append(target["category_id"])
            bboxes.append(target["bbox"])

        if self.transform is not None:
            transformed = self.transform(
                image=np.array(image), bboxes=bboxes, category_ids=category_ids
            )
            
            for idx, tup in enumerate(transformed["bboxes"]):
                transformed["bboxes"][idx] = np.array(tup)

            # Check again after transform because we disregard tiny area bboxes
            if len(transformed["bboxes"]) < 1:
                transformed["bboxes"] = [[1, 1, 1, 1]]
                transformed["category_ids"] = [0]

            targets = {
                "boxes": box_convert(torch.FloatTensor(transformed["bboxes"]), "xywh", "xyxy"),
                "labels": torch.tensor(transformed["category_ids"], dtype=torch.int64),
            }
        
        if self.is_source:
            domain_labels = torch.ones_like(targets['labels'], dtype=torch.uint8)
        else:
            domain_labels = torch.zeros_like(targets['labels'], dtype=torch.uint8)

        targets["is_source"] = domain_labels
        return transformed["image"], targets, file_name

    def __len__(self) -> int:
        return len(self.ids)
