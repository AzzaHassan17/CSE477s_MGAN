import torch
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
 
def build_rpn(in_channels=512, anchor_sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)):
    # Generate anchors
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,            # Anchor sizes per feature map level
        aspect_ratios=aspect_ratios    # Aspect ratios per anchor size
    )
 
    # RPN Head: maps feature map -> objectness logits and bbox deltas
    rpn_head = RPNHead(in_channels, anchor_generator.num_anchors_per_location()[0])
 
    # Full RPN module with training and proposal config
    rpn = RegionProposalNetwork(
        anchor_generator=anchor_generator,
        head=rpn_head,
        fg_iou_thresh=0.7,     # Foreground if IoU > 0.7
        bg_iou_thresh=0.3,     # Background if IoU < 0.3
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_top_n={'training': 2000, 'testing': 1000},
        post_nms_top_n={'training': 2000, 'testing': 1000},
        nms_thresh=0.7         # NMS threshold to remove duplicate boxes
    )
 
    return rpn
