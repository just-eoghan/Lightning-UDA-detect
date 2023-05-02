
import torch
from typing import Dict, List, Tuple, Optional
from torchvision.models.detection.image_list import ImageList
import torch.nn.functional as F

"""
    Contains updated functions to inject into torchvision FRCNN RPN module.
    These additions allow FRCNN to function in the domain adaptive fashion.
"""

def compute_loss(
    self, objectness: torch.Tensor, pred_bbox_deltas: torch.Tensor, labels: List[torch.Tensor], regression_targets: List[torch.Tensor],
    use_pseudo_labeling_weight='none'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        objectness (Tensor)
        pred_bbox_deltas (Tensor)
        labels (List[Tensor])
        regression_targets (List[Tensor])

    Returns:
        objectness_loss (Tensor)
        box_loss (Tensor)
    """

    sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
    sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
    sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

    sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

    objectness = objectness.flatten()

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    box_loss = F.smooth_l1_loss(
        pred_bbox_deltas[sampled_pos_inds],
        regression_targets[sampled_pos_inds],
        beta=1 / 9,
        reduction="sum",
    ) / (sampled_inds.numel())

    if use_pseudo_labeling_weight=='none':
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )
    elif use_pseudo_labeling_weight=='prob':
        weight = F.sigmoid(objectness[sampled_inds])
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds], reduction='none'
        )
        objectness_loss = torch.mean(objectness_loss*weight)

    return objectness_loss, box_loss

def assign_targets_to_anchors(
    self, anchors: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    labels = []
    masks = []
    regression_targets = []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
        is_source = targets_per_image['is_source']
        mask_per_image = is_source.new_ones(1, dtype=torch.uint8) if is_source.any() else is_source.new_zeros(1, dtype=torch.uint8)
        masks.append(mask_per_image)
        if not is_source.any():
            continue
        
        gt_boxes = targets_per_image["boxes"]
        if gt_boxes.numel() == 0:
            # Background image (negative example)
            device = anchors_per_image.device
            matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
            labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
        else:
            match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]
            
            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)
            
            # Background (negative examples)
            bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0.0
            
            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1.0

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                [matched_gt_boxes_per_image], [anchors_per_image]
            )
            regression_targets_per_image = torch.cat(regression_targets_per_image, dim=0)

        regression_targets.append(regression_targets_per_image)
        labels.append(labels_per_image)
    return labels, regression_targets, masks

def forward(
    self,
    images: ImageList,
    features: Dict[str, torch.Tensor],
    targets: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Args:
        images (ImageList): images for which we want to compute the predictions
        features (Dict[str, Tensor]): features computed from the images that are
            used for computing the predictions. Each tensor in the list
            correspond to different feature levels
        targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
            If provided, each element in the dict should contain a field `boxes`,
            with the locations of the ground-truth boxes.
    Returns:
        boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
            image.
        losses (Dict[str, Tensor]): the losses for the model during training. During
            testing, it is an empty dict.
    """
    # RPN uses all feature maps that are available
    features = list(features.values())
    objectness, pred_bbox_deltas = self.head(features)
    anchors = self.anchor_generator(images, features)
    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
    losses = {}
    if self.training:
        assert targets is not None
        labels, regression_targets, masks = self.assign_targets_to_anchors(anchors, targets)
        
        masks = torch.cat(masks, dim=0)
    
        loss_objectness, loss_rpn_box_reg = self.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets
        )

        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
    return boxes, losses

def concat_box_prediction_layers(box_cls: List[torch.Tensor], box_regression: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression

def permute_and_flatten(layer: torch.Tensor, N: int, A: int, C: int, H: int, W: int) -> torch.Tensor:
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer