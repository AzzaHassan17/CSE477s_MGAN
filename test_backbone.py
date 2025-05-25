import torch
from models.mgan_backbone import VGGBackbone
from dataset.citypersons_dataset import CityPersonsDataset
import torchvision.transforms as T

# Define image transforms
transforms = T.Compose([
    T.Resize((1024, 2048)),  # Resize to expected input size
    T.ToTensor(),
])
 
# Initialize dataset
dataset = CityPersonsDataset(
    image_root="dataset/leftImg8bit/train",  # or your correct path
    annotation_path="dataset/annotations/citypersons_train.json",
    transforms=transforms
)
 
# Load one sample
image, target = dataset[0]
print(f"Input image shape: {image.shape}")
 
# Add batch dimension
image = image.unsqueeze(0)  # [1, 3, 1024, 2048]
 
# Initialize backbone
model = VGGBackbone(pretrained=True)
model.eval()
 
# Run forward pass
with torch.no_grad():
    features = model(image)
 
print(f"Feature map shape: {features.shape}")