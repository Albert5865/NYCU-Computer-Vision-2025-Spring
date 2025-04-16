import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
from dataset import DigitDataset
import torchvision.transforms.v2 as transforms
import os
import gc
import json
from tqdm import tqdm
from torch.utils.data import Subset
import random
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import uuid

def evaluate_on_coco(model, dataset, device, save_json_path="val_pred.json"):
    from pathlib import Path
    model.eval()
    results = []

    for idx in tqdm(range(len(dataset)), desc="Validating"):
        img, target = dataset[idx]
        image_id = int(target["image_id"].item())
        with torch.no_grad():
            pred = model([img.to(device)])[0]

        boxes = pred["boxes"].cpu()
        scores = pred["scores"].cpu()
        labels = pred["labels"].cpu()

        for box, score, label in zip(boxes, scores, labels):
            if score < 0.65:
                continue
            x1, y1, x2, y2 = box.tolist()
            results.append({
                "image_id": image_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
                "category_id": int(label)
            })

    with open(save_json_path, "w") as f:
        json.dump(results, f)

    coco_gt = COCO("nycu-hw2-data/valid.json")
    coco_dt = coco_gt.loadRes(save_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]  # stats[0] 是 mAP@IoU=0.50:0.95


def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    full_train_data = DigitDataset(
        image_dir="nycu-hw2-data/train",
        annotation_path="nycu-hw2-data/train.json",
        transforms=transforms.ToTensor()
    )

    val_data = DigitDataset(
        image_dir="nycu-hw2-data/valid",
        annotation_path="nycu-hw2-data/valid.json",
        transforms=transforms.ToTensor()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0005)
   
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)


    num_epochs = 10
    best_map = 0.0
    val_maps = []

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()                                                       

        # Randomly sample 10% of indices
        num_samples = int(0.1 * len(full_train_data))
        sampled_indices = random.sample(range(len(full_train_data)), num_samples)

        # Build Subset dataset and DataLoader
        subset_data = Subset(full_train_data, sampled_indices)
        train_loader = DataLoader(subset_data, batch_size=2, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=2)

        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        model.train()


        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            batch_loss = losses.item()
            total_loss += batch_loss
            pbar.set_postfix(loss=batch_loss)

            del images, targets, loss_dict, losses
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

        # Calculate average loss for the epoch
        epoch_avg = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} | Avg Loss: {epoch_avg:.4f}")

        # Evaluate on validation set
        val_map = evaluate_on_coco(model, val_data, device)
        print(f"Validation mAP: {val_map:.4f}")

        val_maps.append(val_map)

        # Save only if mAP improves
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), f"best_model.pth")
            print("Saved best model with mAP =", best_map)


        # Plot mAP curve
        plt.plot(range(1, len(val_maps) + 1), val_maps, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Validation mAP")
        plt.title("Validation mAP Curve")
        plt.grid(True)
        plt.savefig("map_curve.png")
        plt.close()
        print("mAP curve saved as 'map_curve.png'")

        
        lr_scheduler.step()


if __name__ == "__main__":
    main()
