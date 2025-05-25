import torch
 
def generate_coarse_mask(image_size, visible_boxes):
    """
    Generate a coarse binary mask for visible regions.
    Args:
        image_size (tuple): (height, width) of the image.
        visible_boxes (list of lists): Each sublist contains [x1, y1, x2, y2] for a visible box.
    Returns:
        torch.Tensor: Binary mask of shape (1, H, W).
    """

    H, W = image_size
    mask = torch.zeros((H, W), dtype=torch.float32)
    for box in visible_boxes:
        x1, y1, x2, y2 = map(int, box)
        mask[y1:y2, x1:x2] = 1.0
    return mask.unsqueeze(0)  # Shape: (1, H, W)

 
def compute_occlusion_ratios(full_boxes, visible_boxes):
    """
    Compute occlusion ratios for each pedestrian instance.
    Args:
        full_boxes (list of lists): Each sublist contains [x1, y1, x2, y2] for a full box.
        visible_boxes (list of lists): Each sublist contains [x1, y1, x2, y2] for a visible box.
    Returns:
        torch.Tensor: Tensor of occlusion ratios.
    """

    ratios = []
    for full_box, vis_box in zip(full_boxes, visible_boxes):
        fx1, fy1, fx2, fy2 = full_box
        vx1, vy1, vx2, vy2 = vis_box
        full_area = max((fx2 - fx1), 1) * max((fy2 - fy1), 1)
        vis_area = max((vx2 - vx1), 0) * max((vy2 - vy1), 0)
        ratio = vis_area / full_area
        ratios.append(ratio)
    return torch.tensor(ratios, dtype=torch.float32)

 