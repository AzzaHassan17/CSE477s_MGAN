import torch
import torch.nn as nn
from torchvision.ops import roi_align
from torchvision.models.detection.image_list import ImageList
from models.mgan_backbone import VGGBackbone
from models.rpn import build_rpn
from models.mgam import  MGAM as MaskGuidedAttentionModule
from models.rcnn import RCNNHead
from utils.target_assigner import assign_targets_to_proposals, compute_regression_targets

class MGANDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(MGANDetector, self).__init__()
        self.backbone = VGGBackbone()
        self.rpn = build_rpn()
        self.mgam = MaskGuidedAttentionModule(in_channels=512)
        self.rcnn_head = RCNNHead(in_channels=512, num_classes=num_classes)
        self.roi_output_size = (7, 7)
        self.spatial_scale = 1.0 / 16  # VGG downsampling factor
        self.sampling_ratio = 2 # WTF

    def forward(self, images, targets=None):
        """
        Args:
            images (List[Tensor]): input images [C, H, W]
            targets (List[Dict]): each target dict contains:
                - boxes: Tensor[N, 4] (full)
                - visible_boxes: Tensor[N, 4] (visible)
                - labels: Tensor[N]
        Returns:
            result during evaluation OR loss dict during training
        """

        device = images[0].device
        original_sizes = [img.shape[-2:] for img in images]
        image_list = ImageList(torch.stack(images), original_sizes)
 
        # Feature extraction
        features = self.backbone(image_list.tensors)
        features = {"0": features}
 
        # RPN proposals
        proposals, rpn_losses = self.rpn(image_list, features, targets)


        if self.training:
            # Assign ground truth to proposals
            full_boxes = [t["boxes"] for t in targets]
            vis_boxes = [t["visible_boxes"] for t in targets]
            all_roi_feats_full, all_roi_feats_vis = [], []
            for i, (proposal, full_box, vis_box) in enumerate(zip(proposals, full_boxes, vis_boxes)):
                batch_idx = torch.full((proposal.size(0), 1), i, dtype=torch.float32, device=device)
                rois = torch.cat([batch_idx, proposal], dim=1)
                # RoIAlign for proposals
                roi_feats_full = roi_align(
                    input=features["0"],
                    boxes=rois,
                    output_size=self.roi_output_size,
                    spatial_scale=self.spatial_scale,
                    sampling_ratio=self.sampling_ratio,
                    aligned=True
                )
 
                # RoIAlign for visible ground truth boxes
                gt_idx = torch.full((vis_box.size(0), 1), i, dtype=torch.float32, device=device)
                rois_vis = torch.cat([gt_idx, vis_box], dim=1)
                roi_feats_vis = roi_align(
                    input=features["0"],
                    boxes=rois_vis,
                    output_size=self.roi_output_size,
                    spatial_scale=self.spatial_scale,
                    sampling_ratio=self.sampling_ratio,
                    aligned=True
                )
                all_roi_feats_full.append(roi_feats_full)
                all_roi_feats_vis.append(roi_feats_vis)

            roi_feats_full = torch.cat(all_roi_feats_full, dim=0)
            roi_feats_vis = torch.cat(all_roi_feats_vis, dim=0)

            # Sanity fallback — repeat vis features to match full features (not ideal, for debug only)
            if roi_feats_vis.size(0) != roi_feats_full.size(0):
                repeat_factor = roi_feats_full.size(0) // roi_feats_vis.size(0) + 1
                roi_feats_vis = roi_feats_vis.repeat(repeat_factor, 1, 1, 1)[:roi_feats_full.size(0)]
 
            # Apply MGAM
            fused_feats, mask_preds = self.mgam(roi_feats_full, roi_feats_vis)
 
            # RCNN Head
            class_logits, bbox_deltas_full, bbox_deltas_visible = self.rcnn_head(fused_feats)


 
            # ... existing code to get roi_feats_full, roi_feats_vis, mgam, rcnn head outputs ...

            # Assign ground truth to proposals
            full_boxes = [t["boxes"] for t in targets]
            vis_boxes = [t["visible_boxes"] for t in targets]
            gt_labels = [t["labels"] for t in targets]
            matched_idxs, labels_list = assign_targets_to_proposals(proposals, full_boxes, gt_labels)

            # Compute regression targets for bbox regression
            regression_targets = []
            for i in range(len(proposals)):
                matched_gt_boxes = full_boxes[i][matched_idxs[i].clamp(min=0)]  # clamp to avoid negative indices
                regression_targets_i = compute_regression_targets(proposals[i], matched_gt_boxes)
                regression_targets.append(regression_targets_i)

            return {
                "rpn_loss": rpn_losses,
                "class_logits": class_logits,
                "bbox_deltas_full": bbox_deltas_full,
                "bbox_deltas_visible": bbox_deltas_visible,
                "mask_preds": mask_preds,
                "proposals": proposals,
                "labels": labels_list,
                "regression_targets": regression_targets,
                "matched_idxs": matched_idxs,
            }
        else:
            # Inference
            all_roi_feats_full = []
            for i, proposal in enumerate(proposals):
                batch_idx = torch.full((proposal.size(0), 1), i, dtype=torch.float32, device=device)
                rois = torch.cat([batch_idx, proposal], dim=1)
                roi_feats_full = roi_align(
                    input=features["0"],
                    boxes=rois,
                    output_size=self.roi_output_size,
                    spatial_scale=self.spatial_scale,
                    sampling_ratio=self.sampling_ratio,
                    aligned=True
                )

                all_roi_feats_full.append(roi_feats_full)
 
            roi_feats_full = torch.cat(all_roi_feats_full, dim=0)
            # Use full-body features only at test time
            fused_feats = roi_feats_full
            class_logits, bbox_deltas_full, bbox_deltas_visible = self.rcnn_head(fused_feats)
            full_boxes = [t["boxes"] for t in targets]
            labels_list = []
            matched_idxs = []
        
            # Assign each proposal to a ground-truth box and get labels
            for i, props in enumerate(proposals):
                matched_idx_i, labels_i = assign_targets_to_proposals(props, full_boxes[i], targets[i]["labels"])
                matched_idxs.append(matched_idx_i)
                labels_list.append(labels_i)
        
            # Compute regression targets (bbox deltas) for each proposal matched to GT box
            regression_targets = []
            for i in range(len(proposals)):
                matched_gt_boxes = full_boxes[i][matched_idxs[i]]
                regression_targets_i = compute_regression_targets(proposals[i], matched_gt_boxes)
                regression_targets.append(regression_targets_i)
        
            # --- END NEW ---
        
            return {
            "class_logits": class_logits,
            "bbox_deltas_full": bbox_deltas_full,
            "bbox_deltas_visible": bbox_deltas_visible,
            "mask_preds": mask_preds,
            "proposals": proposals,
        }
                    

