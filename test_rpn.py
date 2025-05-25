import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection.image_list import ImageList
 
from dataset.citypersons_dataset import CityPersonsDataset
from models.mgan_backbone import VGGBackbone
from models.rpn import build_rpn
 
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# Define transforms
transform = transforms.ToTensor()
 
# Load image and annotation
dataset = CityPersonsDataset(
    image_root='dataset/leftImg8bit/train',
    annotation_path='dataset/annotations/citypersons_train.json',
    transforms=transform
)
 
# Get one image-target pair
image, target = dataset[0]
image = image.to(device)
target = {k: v.to(device) for k, v in target.items()}
 
# Wrap in list
original_image_sizes = [image.shape[-2:]]
images = ImageList(torch.stack([image]), original_image_sizes)
targets = [target]
 
# Initialize VGG backbone and RPN
backbone = VGGBackbone().to(device)
rpn = build_rpn().to(device)
 
# Extract features
with torch.no_grad():
    feature_map = backbone(image.unsqueeze(0))  # shape: [1, 512, H, W]
features = {"0": feature_map.to(device)}
 
# Run RPN
rpn.eval()
with torch.no_grad():
    proposals, rpn_losses = rpn(images, features, targets)
 
# Output results
print(f"Number of proposals: {len(proposals[0])}")
print(f"First proposal box: {proposals[0][0]}")