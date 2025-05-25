import torch

from torchvision.ops import roi_align

from models.mgan_backbone import VGGBackbone

from models.rpn import build_rpn

from dataset.citypersons_dataset import CityPersonsDataset

from torchvision import transforms

from torchvision.models.detection.image_list import ImageList
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# Dataset

dataset = CityPersonsDataset(

    image_root='dataset/leftImg8bit/train',

    annotation_path='dataset/annotations/citypersons_train.json',

    transforms=transforms.ToTensor()

)

image, target = dataset[0]

image = image.to(device)

target = {k: v.to(device) for k, v in target.items()}

original_image_sizes = [image.shape[-2:]]

images = ImageList(torch.stack([image]), original_image_sizes)

targets = [target]
 
# Backbone and RPN

backbone = VGGBackbone().to(device)

rpn = build_rpn().to(device)
 
# Feature map

with torch.no_grad():

    feature_map = backbone(image.unsqueeze(0))

    features = {"0": feature_map}

    proposals, _ = rpn(images, features, targets)
 
# Get proposals for batch 0 and build [batch_idx, x1, y1, x2, y2]

proposals = proposals[0]  # [num_proposals, 4]

batch_indices = torch.zeros((proposals.shape[0], 1), device=device)

rois = torch.cat([batch_indices, proposals], dim=1)  # [N, 5]
 
# Apply RoIAlign

roi_features = roi_align(

    input=feature_map,     # [1, 512, H, W]

    boxes=rois,            # [N, 5]

    output_size=(7, 7),    # Per MGAN

    spatial_scale=1.0 / 16,  # VGG16 downsamples 16x

    sampling_ratio=2,

    aligned=True

)

print(f"RoI features shape: {roi_features.shape}")  # [N, 512, 7, 7]

