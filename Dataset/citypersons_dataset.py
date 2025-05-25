import os
import json
import torch
import torchvision.transforms.functional 
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class CityPersonsDataset(Dataset):

    def __init__(self, image_root, annotation_path, transforms=None):

        self.image_root = image_root
        self.coco = COCO(annotation_path)
        self.transforms = transforms
        self.annotations_index = self._build_ann_index()

        # Keep only image_ids that have pedestrian annotations
        self.image_ids = [img_id for img_id in self.coco.imgs.keys() if img_id in self.annotations_index]
 
 

 
    def _build_ann_index(self):
        index = {}
        for ann in self.coco.dataset['annotations']:
            img_id = ann['image_id']
            if img_id not in index:
                index[img_id] = []
            index[img_id].append(ann)
        return index
 
    def __len__(self):
        return len(self.image_ids)
 
    def __getitem__(self, idx):

        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        img_path = os.path.join(self.image_root, file_name)
        image = Image.open(img_path).convert("RGB")
        anns = self.annotations_index[img_id]
        boxes = []
        visible_boxes = []
        labels = []
 
        for ann in anns:
            bbox = ann['bbox']
            vis_bbox = ann.get('visible_bbox', bbox)
            boxes.append(bbox)
            visible_boxes.append(vis_bbox)
            labels.append(1)  # pedestrian

        boxes = torch.tensor(boxes, dtype=torch.float32)
        visible_boxes = torch.tensor(visible_boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,  # full-body bbox
            "visible_boxes": visible_boxes,  # for MGAN
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }
 
        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

 




 