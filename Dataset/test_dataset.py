from citypersons_dataset import CityPersonsDataset

from torchvision import transforms

from torch.utils.data import DataLoader
 
transform = transforms.Compose([

    transforms.ToTensor()

])
 
dataset = CityPersonsDataset(

    image_root="data/leftImg8bit/train",

    annotation_path="data/annotations/citypersons_train.json",

    transforms=transform

)
 
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
 
for images, targets in dataloader:

    print("Image shape:", images[0].shape)

    print("BBoxes:", targets[0]["boxes"].shape)

    print("Visible boxes:", targets[0]["visible_boxes"].shape)

    break

 