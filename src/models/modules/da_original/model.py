from typing import OrderedDict
import torch.nn as nn

import torch
import torchvision
from collections import OrderedDict
from collections import namedtuple
from typing import Tuple, List
import warnings

from torchvision import models
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from src.models.modules.da_injection.da_rpn import assign_targets_to_anchors as da_rpn_assign_targets_to_anchors
from src.models.modules.da_injection.da_rpn import forward as da_rpn_forward
from src.models.modules.da_injection.da_roi import forward as da_roi_forward, select_training_samples as da_roi_select_training_samples
from src.models.modules.da_injection.da_roi import assign_targets_to_proposals as da_assign_targets_to_proposals
from src.models.modules.da_original.da_heads import DomainAdaptationModule
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import RPNHead, AnchorGenerator
from torchvision.ops.poolers import MultiScaleRoIAlign
from src.models.modules.da_original.resnet_head import ResNetHead
from src.models.modules.da_original.frcnn_predictor import FastRCNNPredictor

class DaFRCNN(GeneralizedRCNN):
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
        box_head_res2_out_channels: int,
        ):
        super(GeneralizedRCNN, self).__init__()

        res = torchvision.models.resnet50(pretrained=True)
        # res.children())[:-3] is just resnet with the last 3 layers missing (needed for domain adapt)
        self.backbone = nn.Sequential(*list(res.children())[:-3]).to(torch.device("cuda"))
        self.backbone.out_channels = backbone_out_channels

        anchor_generator = AnchorGenerator(sizes=(tuple([32, 64, 128, 256, 512]),),aspect_ratios=(tuple([0.5, 1.0, 2.0]),))
         
        StageSpec = namedtuple(
            "StageSpec",
            [
                "index",  # Index of the stage, eg 1, 2, ..,. 5
                "block_count",  # Numer of residual blocks in the stage
                "return_features",  # True => return the last feature map from this stage
            ],
        )
        stage = StageSpec(index=4, block_count=3, return_features=False)

        frcnn_base_model = FasterRCNN(
                                self.backbone,
                                rpn_head=RPNHead(backbone_out_channels, 15), 
                                rpn_anchor_generator=anchor_generator,
                                box_batch_size_per_image=256,
                                box_positive_fraction=0.25,
                                box_roi_pool=MultiScaleRoIAlign(featmap_names=['0'], output_size=tuple([14, 14]), sampling_ratio=0),
                                box_head=ResNetHead(
                                    block_module='BottleneckWithFixedBatchNorm',
                                    stages=(stage,),
                                    num_groups=1,
                                    width_per_group=64,
                                    stride_in_1x1=True,
                                    stride_init=None,
                                    res2_out_channels=box_head_res2_out_channels,
                                    dilation=1
                                ),
                                box_nms_thresh=0.5,
                                box_predictor=FastRCNNPredictor(num_classes=num_classes)
                            )

        self.backbone = frcnn_base_model.backbone
        self.transform = frcnn_base_model.transform
        self.rpn = frcnn_base_model.rpn
        self.roi_heads = frcnn_base_model.roi_heads

        # Add DA version of forward for torchvision.detection.rpn
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
        
        self.da_heads = DomainAdaptationModule(res2_out_channels=box_head_res2_out_channels, in_channels=backbone_out_channels)

    def forward(self, images, targets=None):
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
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        da_features = self.backbone(images.tensors)

        if not isinstance(da_features, OrderedDict):
            da_features = OrderedDict({"0": da_features})

        proposals, proposal_losses = self.rpn(images, da_features, targets)
        
        da_losses = {}
        losses = {}
        if not self.training:
            detections, detector_losses = self.roi_heads(da_features, proposals, images.image_sizes, targets)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
        else:
            detections, detector_losses, da_ins_feas, da_ins_labels, da_proposals = self.roi_heads(da_features, proposals, images.image_sizes, targets)
            da_losses = self.da_heads(da_features.values(), da_ins_feas, da_ins_labels, da_proposals, targets)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]


        if self.training:
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(da_losses)
           
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)
