# utils/mask_utils.py
 
import torch
import torch.nn.functional as F

def compute_coarse_mask(proposals, visible_boxes, output_size=(7, 7)):

    """
    Generate a binary mask for each proposal where visible box = 1, background = 0.
    Args:
        proposals (Tensor): [N, 4] full bounding boxes.
        visible_boxes (Tensor): [N, 4] visible bounding boxes.
        output_size (tuple): Size of the mask, usually (7, 7).
    Returns:
        masks: Tensor of shape [N, 1, H, W]
    """

    N = proposals.shape[0]
    H, W = output_size
    device = proposals.device
    masks = torch.zeros((N, 1, H, W), device=device)
    for i in range(N):
        x1, y1, x2, y2 = proposals[i]
        vx1, vy1, vx2, vy2 = visible_boxes[i]
      
        # Map visible box into 7x7 region relative to the proposal box
        px1, py1, px2, py2 = x1, y1, x2, y2
        pw = max(px2 - px1, 1)
        ph = max(py2 - py1, 1)
      
        # Normalize visible box coords into [0, 1]
        rel_x1 = (vx1 - px1) / pw
        rel_y1 = (vy1 - py1) / ph
        rel_x2 = (vx2 - px1) / pw
        rel_y2 = (vy2 - py1) / ph
 
        # Convert to grid coordinates
        grid_x1 = int(rel_x1 * W)
        grid_y1 = int(rel_y1 * H)
        grid_x2 = int(rel_x2 * W)
        grid_y2 = int(rel_y2 * H)
 
        # Clamp
        grid_x1 = max(0, min(grid_x1, W - 1))
        grid_y1 = max(0, min(grid_y1, H - 1))
        grid_x2 = max(grid_x1 + 1, min(grid_x2, W))
        grid_y2 = max(grid_y1 + 1, min(grid_y2, H))
        masks[i, 0, grid_y1:grid_y2, grid_x1:grid_x2] = 1.0
 
    return masks

 
