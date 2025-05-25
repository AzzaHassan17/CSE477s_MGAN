import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import torchvision.ops as ops
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset.citypersons_dataset import CityPersonsDataset
from models.mgan_detector import MGANDetector
from models.mgan_loss import MGANLoss
from tqdm import tqdm

 
def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)
 
 
def nms(boxes, scores, iou_threshold=0.5):
    keep = ops.nms(boxes, scores, iou_threshold)
    return keep
 
def compute_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    inter = (torch.min(box1[:, None, 2], box2[:, 2]) - torch.max(box1[:, None, 0], box2[:, 0])).clamp(min=0) * \
            (torch.min(box1[:, None, 3], box2[:, 3]) - torch.max(box1[:, None, 1], box2[:, 1])).clamp(min=0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter
    return inter / union
 
def compute_precision_recall(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    """
    Compute precision and recall at a fixed IoU threshold
    """
    if len(pred_boxes) == 0:
        return 0, 0  # no preds -> precision=0, recall=0
 
    ious = compute_iou(pred_boxes, gt_boxes)
    matched_gt = set()
    tp = 0
    fp = 0
 
    # Sort preds by score descending
    sorted_inds = torch.argsort(pred_scores, descending=True)
    for i in sorted_inds:
        pred_box = pred_boxes[i]
        max_iou, max_j = 0, -1
        for j in range(gt_boxes.shape[0]):
            if j in matched_gt:
                continue
            iou_val = ious[i, j].item()
            if iou_val > max_iou:
                max_iou = iou_val
                max_j = j
        if max_iou >= iou_thresh:
            tp += 1
            matched_gt.add(max_j)
        else:
            fp += 1
 
    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall
 
 
def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    val_loss = checkpoint.get('val_loss', None)
    print(f"Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")
    return model, optimizer, start_epoch, val_loss
 
 
def evaluate_with_metrics(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0.0
 
    all_precisions = []
    all_recalls = []
 
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating", leave=False):
            images = [img.to(device) for img in images]
            for t in targets:
                for k in ['boxes', 'visible_boxes', 'labels', 'image_id']:
                    t[k] = t[k].to(device)
 
            outputs = model(images, targets)
            losses = criterion(outputs, targets)
            val_loss += losses['loss_total'].item()
 
            # Extract predictions (during eval mode, no mask_preds and RCNN returns logits & bbox_deltas)
            class_logits = outputs['class_logits']  # [N, 2]
            bbox_deltas = outputs['bbox_deltas']    # [N, 4]
            proposals = outputs['proposals']        # List of proposals per image
 
            batch_size = len(images)
            for i in range(batch_size):
                scores = torch.softmax(class_logits[i], dim=1)[:, 1]  # Pedestrian class prob
                boxes = proposals[i]
 
                # Apply NMS
                keep = nms(boxes, scores, iou_threshold=0.5)
                boxes = boxes[keep]
                scores = scores[keep]
 
                gt_boxes = targets[i]['boxes']
 
                precision, recall = compute_precision_recall(boxes, scores, gt_boxes)
                all_precisions.append(precision)
                all_recalls.append(recall)
 
    avg_val_loss = val_loss / len(val_loader)
    mean_precision = np.mean(all_precisions) if all_precisions else 0
    mean_recall = np.mean(all_recalls) if all_recalls else 0
 
    # Approximate miss rate (MR) = 1 - recall (rough approx, real MR calc is more complex)
    mean_mr = 1 - mean_recall
 
    print(f"Validation Loss: {avg_val_loss:.4f}, Precision: {mean_precision:.4f}, Recall: {mean_recall:.4f}, Approx. MR: {mean_mr:.4f}")
 
    return avg_val_loss, mean_precision, mean_recall, mean_mr
 
 
def main():
    import torch
    print("CUDA Available:", torch.cuda.is_available())
    print("Torch CUDA Version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    print("Current device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")




    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths (update accordingly)
    train_image_root = "C:/Users/7ekmaz/OneDrive/Desktop/CSE/Senior_1/2nd_Term/Deep/Project/MGAN/dataset/leftImg8bit/train"
    train_ann_path = "C:/Users/7ekmaz/OneDrive/Desktop/CSE/Senior_1/2nd_Term/Deep/Project/MGAN/dataset/annotations/citypersons_train.json"
    val_image_root = "C:/Users/7ekmaz/OneDrive/Desktop/CSE/Senior_1/2nd_Term/Deep/Project/MGAN/dataset/leftImg8bit/val"
    val_ann_path = "C:/Users/7ekmaz/OneDrive/Desktop/CSE/Senior_1/2nd_Term/Deep/Project/MGAN/dataset/annotations/citypersons_val.json"

    # Transforms
    train_transforms = T.Compose([
    T.Resize((256, 512)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),])
    val_transforms = T.Compose([T.ToTensor()])
 
    # Datasets and loaders
    train_dataset = CityPersonsDataset(train_image_root, train_ann_path, train_transforms)
    val_dataset = CityPersonsDataset(val_image_root, val_ann_path, val_transforms)
 
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
 
    # Model and loss
    model = MGANDetector(num_classes=2).to(device)
    print("Model is on:", next(model.parameters()).device)

    criterion = MGANLoss(alpha=0.5, beta=1.0)
 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8], gamma=0.1)
 
    num_epochs = 11
    best_val_loss = float('inf')
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
   
    start_epoch = 0
    best_val_loss = float('inf')
    # If resuming training from checkpoint
    resume_path = "checkpoints/checkpoint.pth"
    if resume_path and os.path.exists(resume_path):
        model, optimizer, start_epoch, best_val_loss = load_checkpoint(model, optimizer, resume_path, device)
 
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images = [img.to(device) for img in images]
            print("First image tensor device:", images[0].device)
            for t in targets:
                for k in ['boxes', 'visible_boxes', 'labels', 'image_id']:
                    t[k] = t[k].to(device)
 
            optimizer.zero_grad()
            outputs = model(images, targets)
            losses = criterion(outputs, targets)
            loss = losses['loss_total']
 
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            epoch_loss += loss.item()
 
        lr_scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss, precision, recall, miss_rate = evaluate_with_metrics(model, criterion, val_loader, device)
 
        print(f"Epoch [{epoch+1}/{num_epochs}] === , Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, miss rate: {miss_rate:.4f}")
 
        # Save checkpoint if val loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"Saved best model checkpoint at epoch {epoch+1}")
 
if __name__ == "__main__":
    main()
 
 
