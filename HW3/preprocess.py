import os
import torch
import numpy as np
import tifffile
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


class InstanceSegmentationDataset(Dataset):
    """
    Custom Dataset for Instance Segmentation task with data augmentation.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.image_ids = []
        self.uuid_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        for idx, uuid in enumerate(self.uuid_dirs):
            img_path = os.path.join(root_dir, uuid, "image.tif")
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.image_ids.append(idx)

        # Define augmentation pipeline
        self.augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(p=0.2),
            A.HueSaturationValue(p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_id = self.image_ids[idx]
        uuid = os.path.basename(os.path.dirname(img_path))

        # Read image
        image = tifffile.imread(img_path)
        image = np.array(Image.fromarray(image).convert("RGB"))

        masks, boxes, labels = [], [], []

        for class_id in range(1, 5):
            mask_path = os.path.join(self.root_dir, uuid, f"class{class_id}.tif")
            if os.path.exists(mask_path):
                mask = tifffile.imread(mask_path)
                instance_ids = np.unique(mask)
                instance_ids = instance_ids[instance_ids > 0]

                for inst_id in instance_ids:
                    instance_mask = (mask == inst_id).astype(np.uint8)
                    if instance_mask.sum() == 0:
                        continue
                    y_indices, x_indices = np.where(instance_mask)
                    y_min, y_max = y_indices.min(), y_indices.max()
                    x_min, x_max = x_indices.min(), x_indices.max()
                    if y_max > y_min and x_max > x_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        masks.append(instance_mask)
                        labels.append(class_id)

        # Prepare masks for augmentation
        if masks:
            masks = np.stack(masks, axis=0)  # Shape: (num_instances, height, width)
        else:
            masks = np.empty((0, image.shape[0], image.shape[1]), dtype=np.uint8)

        # Apply augmentation
        transformed = self.augmentation_pipeline(image=image, masks=masks, bboxes=boxes, class_labels=labels)
        image = transformed['image']
        masks = transformed['masks']
        boxes = transformed['bboxes']
        labels = transformed['class_labels']

        # Convert image to tensor
        image_tensor = transforms.ToTensor()(image)

        # Convert masks and boxes to tensors
        if boxes:
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
                masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id])
        }

        return image_tensor, target