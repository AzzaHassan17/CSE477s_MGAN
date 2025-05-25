import os
import json
import glob
from tqdm import tqdm
import argparse
 
def convert_citypersons_split(gt_dir, img_dir, split):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "pedestrian"}]
    }
 
    img_id = 0
    ann_id = 0
    cities = os.listdir(os.path.join(gt_dir, split))
    
    for city in tqdm(cities, desc=f"Processing {split}"):
        ann_files = sorted(glob.glob(os.path.join(gt_dir, split, city, '*_gtBboxCityPersons.json')))
        for ann_path in ann_files:
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
 
            # Derive image name
            base_name = os.path.basename(ann_path).replace('_gtBboxCityPersons.json', '')
            file_name = f"{city}/{base_name}_leftImg8bit.png"
            file_path = file_name
            width = ann_data['imgWidth']
            height = ann_data['imgHeight']
            coco["images"].append({
                "file_name": file_path,
                "height": height,
                "width": width,
                "id": img_id
            })
 
            for obj in ann_data.get('objects', []):

                if obj['label'] != 'pedestrian':

                    continue  # Ignore non-pedestrian entries
 
                bbox = obj['bbox']

                vis_bbox = obj.get('bboxVis', bbox)
 
                coco["annotations"].append({

                    "id": ann_id,

                    "image_id": img_id,

                    "category_id": 1,

                    "bbox": bbox,  # [x, y, w, h]

                    "visible_bbox": vis_bbox,  # Custom field for MGAN

                    "area": bbox[2] * bbox[3],

                    "iscrowd": 0

                })

                ann_id += 1
 
            img_id += 1
 
    return coco

 
 
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", default="data/gtBboxCityPersons", help="Path to ground truth annotation directory")
    parser.add_argument("--img_dir", default="data/rightImg8bit", help="Path to image directory")
    parser.add_argument("--output_dir", default="data/annotations", help="Where to save converted COCO-style annotations")
    args = parser.parse_args()
 
    os.makedirs(args.output_dir, exist_ok=True)
 
    for split in ['train', 'val']:

        coco_ann = convert_citypersons_split(args.gt_dir, args.img_dir, split)

        output_file = os.path.join(args.output_dir, f"citypersons_{split}.json")

        with open(output_file, 'w') as f:

            json.dump(coco_ann, f)

        print(f"Saved {split} annotations to {output_file}")
 
if __name__ == "__main__":

    main()

 