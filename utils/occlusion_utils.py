# utils/occlusion_utils.py
 
import torch
 
def compute_occlusion_ratios(proposals, visible_boxes):

    """

    Compute occlusion ratio as visible_area / full_area for each proposal.
 
    Args:

        proposals (Tensor): [N, 4]

        visible_boxes (Tensor): [N, 4]
 
    Returns:

        ratios (Tensor): [N]

    """

    px1, py1, px2, py2 = proposals[:, 0], proposals[:, 1], proposals[:, 2], proposals[:, 3]

    vx1, vy1, vx2, vy2 = visible_boxes[:, 0], visible_boxes[:, 1], visible_boxes[:, 2], visible_boxes[:, 3]
 
    full_area = (px2 - px1).clamp(min=1) * (py2 - py1).clamp(min=1)

    vis_area = (vx2 - vx1).clamp(min=0) * (vy2 - vy1).clamp(min=0)
 
    ratio = vis_area / full_area

    return ratio

 