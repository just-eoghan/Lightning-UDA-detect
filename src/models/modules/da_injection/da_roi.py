from operator import is_
import torch
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops

"""
    Contains updated functions for standard FRCNN ROI module.

    These additions allow FRCNN to function in the domain adaptive fashion
"""


# Updated version of select_training_samples to handle DA 
def select_training_samples(
        self,
        proposals: List[torch.Tensor], 
        targets: Optional[List[Dict[str, torch.Tensor]]],
        da: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        gt_source = [t["is_source"] for t in targets]
        for idx, (proposal, gt_box) in enumerate(zip(proposals, gt_boxes)):
            if proposal.nelement() == 0:
                proposals[idx] = gt_box

        # get matching gt indices for each proposal
        labels, matched_targets, domain_labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, gt_source, sample_for_da=da)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            labels_per_image = labels[img_idx][img_sampled_inds]
            mts_per_image = matched_targets[img_idx][img_sampled_inds]
            domain_ls_per_image = domain_labels[img_idx][img_sampled_inds]
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            labels[img_idx] = labels_per_image
            matched_targets[img_idx] = mts_per_image
            domain_labels[img_idx] = domain_ls_per_image
            proposals[img_idx] = proposals_per_image

        regression_targets = self.box_coder.encode(matched_targets, proposals)

        return proposals, labels, regression_targets, domain_labels

def forward(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, "target keypoints must of float type"   

        if self.training:
            proposals, labels, regression_targets, domain_labels = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
        
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        
        # Break out of forward here for a validation call
        if not self.training:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
            return result, losses

        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg, _ = da_fastrcnn_loss(class_logits, box_regression, labels, regression_targets, domain_labels)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
            
            da_proposals, _, _, _ = self.select_training_samples(proposals, targets, da=True)
            da_box_features = self.box_roi_pool(features, da_proposals, image_shapes)
            da_box_features = self.box_head(da_box_features)
            da_class_logits, da_box_regression = self.box_predictor(da_box_features)
        
            _, _, da_ins_labels = da_fastrcnn_loss(da_class_logits, da_box_regression, labels, regression_targets, domain_labels)               

        return (
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
            box_features,
            da_ins_labels,
            da_proposals
        )

def da_fastrcnn_loss(
    class_logits: torch.Tensor,
    box_regression: torch.Tensor, 
    labels: List[torch.Tensor], 
    regression_targets: List[torch.Tensor],
    domain_labels: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes a domain adaptation compatible loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
        domain_labels (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    domain_masks = torch.cat(domain_labels, dim=0)
    
    class_logits = class_logits[domain_masks, :]
    box_regression = box_regression[domain_masks, :]
    labels = labels[domain_masks]
    regression_targets = regression_targets[domain_masks, :]

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    # N, num_classes = class_logits.shape
    # box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    # box_regression[sampled_pos_inds_subset, labels_pos],
    
    map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=class_logits.device)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset[:, None], map_inds],
        regression_targets[sampled_pos_inds_subset],
        size_average=False,
        beta=1
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss, domain_masks

def assign_targets_to_proposals(
    self,
    proposals: List[torch.Tensor], 
    gt_boxes: List[torch.Tensor], 
    gt_labels: List[torch.Tensor],
    gt_source: List[torch.Tensor],
    sample_for_da: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    matched_tars = []
    labels = []
    domain_labels = []
    for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, is_source in zip(proposals, gt_boxes, gt_labels, gt_source):
        if gt_boxes_in_image.numel() == 0:
            # Background image
            device = proposals_in_image.device
            clamped_matched_idxs_in_image = torch.zeros(
                (proposals_in_image.shape[0],), dtype=torch.int64, device=device
            )
            labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
        else:
            #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
            
            # Equivilant of match_targets_to_proposals in krumo
            if proposals_in_image.nelement() == 0:
                proposals_in_image = gt_boxes_in_image
            match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
            matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)
            clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
            matched_targets = gt_boxes_in_image[clamped_matched_idxs_in_image]
            labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
            if not is_source.any():
                labels_in_image = gt_labels_in_image[matched_idxs_in_image]
            
            labels_in_image = labels_in_image.to(dtype=torch.int64)
            # Label background (below the low threshold)
            bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_in_image[bg_inds] = 0
            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            domain_label = torch.ones_like(labels_in_image, dtype=torch.uint8) if is_source.any() else torch.zeros_like(labels_in_image, dtype=torch.uint8)
            domain_labels.append(domain_label)

            if not is_source.any():
                labels_in_image[:] = 0
            if sample_for_da:
                labels_in_image[:] = 0
        
        matched_tars.append(matched_targets)
        labels.append(labels_in_image)
    return labels, matched_tars, domain_labels