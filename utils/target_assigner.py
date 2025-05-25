import torch
from torchvision.ops import box_iou
import torch
from torchvision.ops import box_iou

def assign_targets_to_proposals(proposals, gt_boxes, gt_labels, iou_threshold=0.5):
    matched_idxs = []
    labels = []
    for props_per_img, gt_per_img, lbl_per_img in zip(proposals, gt_boxes, gt_labels):
        if gt_per_img.numel() == 0:
            matched_idxs.append(torch.full((props_per_img.shape[0],), -1, dtype=torch.int64))
            labels.append(torch.full((props_per_img.shape[0],), 0, dtype=torch.int64))  # all background
            continue
        ious = box_iou(props_per_img, gt_per_img)  # [num_props, num_gt]
        max_ious, matched_gt_idx = ious.max(dim=1)
        labels_per_img = lbl_per_img[matched_gt_idx].clone()
        labels_per_img[max_ious == 0] = -1  # ignore
        labels_per_img[(max_ious < iou_threshold) & (max_ious > 0)] = 0  # background
        # Clamp to {1} for all foregrounds above 1
        labels_per_img[(labels_per_img > 1)] = 1
        labels_per_img = labels_per_img.to(torch.long)
        matched_idxs.append(matched_gt_idx)
        labels.append(labels_per_img)
    return matched_idxs, labels

def compute_regression_targets(proposals, matched_gt_boxes):
    wx, wy, ww, wh = (1., 1., 1., 1.)
    proposals = proposals.to(dtype=torch.float32)
    matched = matched_gt_boxes.to(dtype=torch.float32)
    px, py, pw, ph = get_box_center(proposals)
    gx, gy, gw, gh = get_box_center(matched)
    dx = wx * (gx - px) / pw
    dy = wy * (gy - py) / ph
    dw = ww * torch.log(gw / pw)
    dh = wh * torch.log(gh / ph)
    return torch.stack((dx, dy, dw, dh), dim=1)


def get_box_center(boxes):
    x1, y1, x2, y2 = boxes.unbind(1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h
 