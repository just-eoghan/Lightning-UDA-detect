from typing import OrderedDict
import torch.nn as nn

import torch
from collections import OrderedDict
from typing import Tuple, List
import warnings

import torchvision

from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights

from src.models.modules.da_mic_injection.da_rpn import assign_targets_to_anchors as da_rpn_assign_targets_to_anchors
from src.models.modules.da_mic_injection.da_rpn import forward as da_rpn_forward
from src.models.modules.da_mic_injection.da_rpn import compute_loss as da_rpn_compute_loss
from src.models.modules.da_mic_injection.da_roi import forward as da_roi_forward, select_training_samples as da_roi_select_training_samples
from src.models.modules.da_mic_injection.da_roi import assign_targets_to_proposals as da_assign_targets_to_proposals
from src.models.modules.da_scale_aware.da_heads import SaDomainAdaptationModule
from torchvision.models.resnet import ResNet50_Weights

def normalize_image(image):
    min_val = torch.min(image)
    max_val = torch.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

class MicDaFRCNN(GeneralizedRCNN):
    """
        Module for domain adapative faster rcnn
        Three parts
        - backbone
        - rpn
        - heads
    """

    def __init__(
            self,
            num_classes: int,
            backbone_out_channels: int,
            consit_weight: float,
            img_grl_weight: float,
            ins_grl_weight: float
        ):
        super(GeneralizedRCNN, self).__init__()

        kwargs = dict(box_batch_size_per_image = 256)
        frcnn_base_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
            num_classes=num_classes, 
            **kwargs
        )

        self.backbone = frcnn_base_model.backbone
        self.transform = frcnn_base_model.transform
        self.rpn = frcnn_base_model.rpn
        self.roi_heads = frcnn_base_model.roi_heads

        # Add DA version of forward for torchvision.detection.rpn
        bound_method = da_rpn_compute_loss.__get__(self.rpn, self.rpn.__class__)
        setattr(self.rpn, 'compute_loss', bound_method)
        bound_method = da_rpn_assign_targets_to_anchors.__get__(self.rpn, self.rpn.__class__)
        setattr(self.rpn, 'assign_targets_to_anchors', bound_method)
        bound_method = da_rpn_forward.__get__(self.rpn, self.rpn.__class__)
        setattr(self.rpn, 'forward', bound_method)

        # Add DA versions of forward and select_training_samples for torchvision.detection.roi_heads
        bound_method = da_roi_forward.__get__(self.roi_heads, self.roi_heads.__class__)
        setattr(self.roi_heads, 'forward', bound_method)
        bound_method = da_roi_select_training_samples.__get__(self.roi_heads, self.roi_heads.__class__)
        setattr(self.roi_heads, 'select_training_samples', bound_method)
        bound_method = da_assign_targets_to_proposals.__get__(self.roi_heads, self.roi_heads.__class__)
        setattr(self.roi_heads, 'assign_targets_to_proposals', bound_method)
        
        self.da_heads = SaDomainAdaptationModule(
            in_channels=backbone_out_channels,
            consit_weight=consit_weight,
            img_grl_weight=img_grl_weight,
            ins_grl_weight=ins_grl_weight
        )

    def forward(self, images, targets=None, use_pseudo_labeling_weight='none', with_DA_ON=True):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")
                else:
                    raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        # Loop
        ims = []
        for im in images.tensors:
            ims.append(normalize_image(im))

        da_features = self.backbone(torch.stack(ims))

        if not isinstance(da_features, OrderedDict):
            da_features = OrderedDict({"0": da_features})

        proposals, proposal_losses = self.rpn(images, da_features, targets, use_pseudo_labeling_weight=use_pseudo_labeling_weight)
        
        da_losses = {}
        losses = {}
        if not self.training:
            detections, detector_losses = self.roi_heads(da_features, proposals, images.image_sizes, targets, use_pseudo_labeling_weight=use_pseudo_labeling_weight)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
        else:
            detections, detector_losses, da_ins_feas, da_ins_labels, da_proposals = self.roi_heads(da_features, proposals, images.image_sizes, targets)
            # da_features
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
            if self.da_heads and with_DA_ON:
                da_losses = self.da_heads(da_features.values(), da_ins_feas, da_ins_labels, da_proposals, targets)

        if self.training:
            losses.update(detector_losses)
            losses.update(proposal_losses)
            if with_DA_ON:
                losses.update(da_losses)
           
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)
