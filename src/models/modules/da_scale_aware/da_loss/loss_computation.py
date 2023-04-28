"""
This file contains specific functions for computing losses on the da_heads
file
"""

import torch
from torch import nn
from torch.nn import functional as F
from src.models.modules.da_scale_aware.da_loss.pooler import Pooler

class SaDALossComputation(object):
    """
    This class computes the DA loss.
    """

    def __init__(self):
        resolution = 7
        self.avgpool = nn.AvgPool2d(kernel_size=resolution, stride=resolution)
        pooler = Pooler(
            output_size=(7, 7),
            scales=(0.25, 0.125, 0.0625, 0.03125),
            sampling_ratio=2,
        )
        self.pooler = pooler


    def prepare_masks(self, targets):
        masks = []
        for targets_per_image in targets:
            is_source = targets_per_image['is_source']
            mask_per_image = is_source.new_ones(1, dtype=torch.uint8) if is_source.any() else is_source.new_zeros(1, dtype=torch.uint8)
            masks.append(mask_per_image)
        return masks

    def __call__(self, proposals, da_img, da_ins, da_img_consist, da_ins_consist, da_ins_labels, targets):
        """
        Arguments:
            da_proposals
            da_img (list[Tensor])
            da_img_consist (list[Tensor])
            da_ins (Tensor)
            da_ins_consist (Tensor)
            da_ins_labels (Tensor)
            targets (list[BoxList])

        Returns:
            da_img_loss (Tensor)
            da_ins_loss (Tensor)
            da_consist_loss (Tensor)
        """

        masks = self.prepare_masks(targets)
        masks = torch.cat(masks, dim=0)

        _, _, H, W = da_img[0].shape
        up_sample = nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)
        upsampled_loss = []
        for i, feat in enumerate(da_img):
            feat = da_img[i]
            feat = up_sample(feat)
            da_img_label_per_level = torch.zeros_like(feat, dtype=torch.float32)
            da_img_label_per_level[masks, :] = 1
            lv_loss = F.binary_cross_entropy_with_logits\
                (feat, da_img_label_per_level, reduction='none')
            upsampled_loss.append(lv_loss)

        da_img_loss = torch.stack(upsampled_loss)
        da_img_loss = da_img_loss.mean()

        da_ins_loss = F.binary_cross_entropy_with_logits(
            torch.squeeze(da_ins), da_ins_labels.type(torch.cuda.FloatTensor)
        )

        da_img_rois_probs = self.pooler(da_img_consist, proposals)
        da_img_rois_probs_pool = self.avgpool(da_img_rois_probs)
        da_img_rois_probs_pool = da_img_rois_probs_pool.view(da_img_rois_probs_pool.size(0), -1)
        da_consist_loss = F.l1_loss(da_img_rois_probs_pool, da_ins_consist)

        return da_img_loss, da_ins_loss, da_consist_loss