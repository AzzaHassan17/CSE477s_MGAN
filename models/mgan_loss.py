import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import fastrcnn_loss
from utils.mask_utils import compute_coarse_mask
from utils.occlusion_utils import compute_occlusion_ratios

class MGANLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
 
    def forward(self, predictions, targets):
        class_logits = predictions['class_logits']      # list of [num_props_i, num_classes]
        #bbox_preds = predictions['bbox_deltas']         # list of [num_props_i, 4]
        bbox_preds_full = predictions['bbox_deltas_full']  # list of [num_props_i, 4]
        bbox_preds_visible = predictions['bbox_deltas_visible']  # list of [num_props_i, 4]
        mask_logits = predictions['mask_preds']         # [sum(num_props_i), 1, 28, 28]
        proposals = predictions['proposals']            # list of [num_props_i, 4]
        labels_list = predictions['labels']             # list of [num_props_i]
        regression_targets = predictions['regression_targets']   # list of [num_props_i, 4]
        matched_idxs = predictions['matched_idxs']      # list of [num_props_i]
 
        # === Filter out images with all -1 labels ===
        valid_indices = [i for i, labels in enumerate(labels_list) if (labels >= 0).sum() > 0]
        # If all proposals are ignored, return zero loss
        if not valid_indices:
            device = class_logits[0].device
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                'loss_total': zero,
                'loss_rpn_rcnn': zero,
                'loss_mask': zero,
                'loss_occlusion': zero
            }
 
        # Concatenate proposals for valid images
        filtered_class_logits = torch.cat([class_logits[i] for i in valid_indices], dim=0)  # [N, num_classes]
        #filtered_bbox_preds = torch.cat([bbox_preds[i] for i in valid_indices], dim=0)      # [N, 4]
        filtered_bbox_preds_full = torch.cat([bbox_preds_full[i] for i in valid_indices], dim=0)
        filtered_bbox_preds_visible = torch.cat([bbox_preds_visible[i] for i in valid_indices], dim=0)
        filtered_labels = torch.cat([labels_list[i] for i in valid_indices], dim=0)         # [N]
        filtered_regression_targets = torch.cat([regression_targets[i] for i in valid_indices], dim=0)  # [N, 4]

        # Debug: check label range
        unique_vals = filtered_labels.unique(sorted=True)
        print(f"[DEBUG] Filtered labels unique values: {unique_vals.tolist()}")
        if (filtered_labels >= 2).any() or (filtered_labels < -1).any():
            print(f"[ERROR] Out-of-range value detected in filtered_labels: {filtered_labels}")
            raise ValueError("Invalid label value detected!")

        # === RCNN classification and bbox loss ===
        l0_cls, l0_bbox = fastrcnn_loss(
            filtered_class_logits,
            #filtered_bbox_preds,
            #filtered_bbox_preds_full,
            filtered_bbox_preds_visible,
            [filtered_labels],
            [filtered_regression_targets]
        )

        total_l0 = l0_cls + l0_bbox
        total_lmask = 0.0
        total_locc = 0.0
 
        # Compute per-image mask and occlusion losses
        start_idx = 0
        for i in range(len(targets)):
            num_props = labels_list[i].shape[0]
            end_idx = start_idx + num_props
            labels_img = labels_list[i]
            matched_idxs_img = matched_idxs[i]
            mask_logits_img = mask_logits[start_idx:end_idx].squeeze(1)  # [N, 28, 28]
            class_logits_img = class_logits[i]
            pos_inds = labels_img > 0
            if pos_inds.sum() == 0:
                start_idx = end_idx
                continue
 
            gt_boxes = targets[i]['boxes']
            vis_boxes = targets[i]['visible_boxes']
            pos_matched_idxs = matched_idxs_img[pos_inds]
            matched_full_boxes = gt_boxes[pos_matched_idxs]
            matched_vis_boxes = vis_boxes[pos_matched_idxs]
 
            # === Mask Loss ===
            pred_masks = torch.sigmoid(mask_logits_img[pos_inds])  # [N_pos, 28, 28]
            coarse_masks = compute_coarse_mask(matched_full_boxes, matched_vis_boxes, mask_size=28)  # [N_pos, 28, 28]
            lmask = self.bce_loss(pred_masks, coarse_masks)
            total_lmask += lmask
 
            # === Occlusion-sensitive classification loss ===
            occlusion_ratios = compute_occlusion_ratios(matched_full_boxes, matched_vis_boxes)  # [N_pos]
            selected_logits = class_logits_img[pos_inds]
            selected_labels = labels_img[pos_inds]
            ce = self.ce_loss(selected_logits, selected_labels)  # [N_pos]
            locc = (ce * occlusion_ratios).mean()
            total_locc += locc
            start_idx = end_idx

        total_loss = total_l0 + self.alpha * total_lmask + self.beta * total_locc
        return {
            'loss_total': total_loss,
            'loss_rpn_rcnn': total_l0,
            'loss_mask': total_lmask,
            'loss_occlusion': total_locc
        }
 