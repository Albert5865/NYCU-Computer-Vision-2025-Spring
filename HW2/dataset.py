import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from PIL import Image
import json
import os

class DigitDataset(Dataset):
    def __init__(self, image_dir, annotation_path, transforms=None, subset_ratio=1.0):
        self.image_dir = image_dir
        self.transforms = transforms
        with open(annotation_path) as f:
            annotations = json.load(f)

        self.images = {img["id"]: img for img in annotations["images"]}
        self.annotations = {}
        for anno in annotations["annotations"]:
            img_id = anno["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(anno)

        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.image_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        annos = self.annotations.get(img_id, [])
        boxes = []
        labels = []

        for anno in annos:
            x, y, w, h = anno["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(anno["category_id"])  # starts from 1

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target
