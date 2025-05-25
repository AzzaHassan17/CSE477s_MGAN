# CSE477s_MGAN 🧠🚶‍♂️  
A PyTorch reimplementation of the **Mask-Guided Attention Network (MGAN)** for occluded pedestrian detection — developed from scratch as a course project for CSE477s (Spring 2025).

This repository replicates the main ideas of the MGAN (ICCV 2019) paper, using the **CityPersons** dataset and focuses on interpretability, modularity, and correctness without depending on third-party detection frameworks like MMDetection.

---

## 📁 Project Structure

```
MGAN-Implementation/
├── datasets/
│   ├── annotations/                   # COCO-style annotations
│   ├── gtBboxCityPersons/            # Ground truth bounding boxes
│   ├── leftImg8bit/                  # Cityscapes images
│   ├── citypersons_dataset.py        # Dataset loader
│   └── convert_citypersons_to_coco.py# Converts to COCO format
│
├── models/
│   ├── mgan_detector.py              # Main model interface
│   ├── mgan_backbone.py              # Feature extractor (VGG or FPN)
│   ├── rpn.py                        # Region Proposal Network
│   ├── rcnn.py                       # RCNN heads for full and visible boxes
│   ├── coarse_seg.py                 # Visible region supervision
│   ├── mgam.py                       # Mask-Guided Attention Module
│   └── mgan_loss.py                  # Combined MGAN loss
│
├── utils/
│   ├── mask_utils.py                 # Utilities for mask processing
│   └── occlusion_utils.py           # Occlusion ratio estimation
|   └── target_assigner.py            # Assign targets to labels

├── train.py                          # Main training script
└── test_*.py                         # Unit test scripts for modules
```

---

## ✨ Features

- ✅ Complete modular PyTorch implementation of MGAN  
- 🔁 Dual-branch prediction for full-body and visible-body bounding boxes  
- 🎯 Mask-Guided Attention Module (MGAM) to enhance detection under occlusion  
- 🔍 Weak supervision using visible bounding boxes  
- 🧮 Custom loss combining classification, regression, and attention supervision  
- 🔬 Component-wise testing and debugging utilities  

---

## 🗂 Dataset Setup

MGAN is trained on the **CityPersons dataset**, which is based on Cityscapes.

### 🔧 Preparation Steps

1. Download from the official CityPersons site:
   - `leftImg8bit/`
   - `gtBboxCityPersons/`
   - Annotation JSON files

2. Convert annotations to COCO format:
   ```bash
   python datasets/convert_citypersons_to_coco.py
   ```

3. This will generate:
   - `citypersons_train.json`
   - `citypersons_val.json`  
   inside `datasets/annotations/`.

---

## 🧪 Module Testing

Individual test scripts used during development:

- `test_backbone.py`
- `test_rpn.py`
- `test_rcnn.py`
- `test_mask_module.py`
- `test_loss.py`

These verify shape correctness, module output, and loss values.

---

## ⚠️ Limitations

### 🚧 Training Challenges
- ⚠️ Unstable convergence during model training, especially under heavy occlusion.
- 📉 Difficulty in reproducing qualitative results shown in the original MGAN paper.

### 📦 Data Issues
- ❓ Uncertainty regarding the exact version or preprocessing of the CityPersons dataset used by the original authors.

### ⏱️ Time Constraints
- ⏳ Limited project duration restricted thorough hyperparameter tuning and extensive evaluation.


---

## ✅ Key Achievements

- 🔨 Rebuilt MGAN pipeline from scratch in PyTorch
- 🧠 Captured the core innovations of the original paper
- 🗂️ Fully modular and easy-to-read codebase
- 🧪 Verified each module with isolated tests

---

## 🧑‍💻 Contributors

Developed as part of the **CSE477s Spring 2025** course:

- **Kareem Wael Elhamy**  
  `ID: 2100631`

- **Azza Hassan Said**  
  `ID: 2101808`

---

## 📚 Citation

If using this codebase, cite the original MGAN paper:

```bibtex
@inproceedings{MGAN_ICCV19,
  title={Occluded Pedestrian Detection Through Guided Attention in CNNs},
  author={Zhang, Shanshan and Chi, Cheng and Yao, Zhen and Lei, Hongyu and Li, Stan Z.},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2019}
}
```

---

## 📄 License

For educational and research use only. Contact contributors for reuse or questions.
