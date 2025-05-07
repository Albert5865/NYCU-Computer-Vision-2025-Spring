import os
import gc
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import timm
import tifffile
import albumentations as A
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image
from model import CustomBackbone
from preprocess import InstanceSegmentationDataset


def create_model(num_classes=5):
    """
    Create and return a Mask R-CNN model using a custom backbone.
    """
    model = timm.create_model('resnetrs200.tf_in1k', pretrained=True, features_only=False)
    backbone = CustomBackbone(model)
    backbone_with_fpn = models.detection.backbone_utils.BackboneWithFPN(
        backbone,
        return_layers={"layer1": "1", "layer2": "2", "layer3": "3", "layer4": "4"},
        in_channels_list=backbone.out_channels,
        out_channels=256
    )
    anchor = AnchorGenerator(
        sizes=((8, 16), (16, 32), (32, 64), (64, 128)), 
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )
    box_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['1', '2', '3', '4'], output_size=7, sampling_ratio=2)
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['1', '2', '3', '4'], output_size=14, sampling_ratio=2)

    model = MaskRCNN(backbone_with_fpn, 
                     num_classes=num_classes, 
                     rpn_anchor=anchor,
                     box_roi_pool=box_roi_pooler, 
                     mask_roi_pool=mask_roi_pooler, 
                     box_detections_per_image=1000)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


# Core training logic
def train_model():
    """
    Executes the training process for the Mask R-CNN model with early stopping.
    """
    # Determine device
    computing_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {computing_device}")
    
    # Initialize model and move to device
    trained_model = create_model(num_classes=5).to(computing_device)
    
    # Set up optimizer and scaler
    training_optimizer = AdamW([param for param in trained_model.parameters() if param.requires_grad], 
                              lr=1e-4, weight_decay=5e-4)
    precision_scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None

    # Early stopping setup
    lowest_loss = float('inf')
    stop_patience = 0
    max_patience = 10
    epoch_losses = []

    trained_model.train()

    # Prepare dataset
    complete_dataset = InstanceSegmentationDataset(root_dir="hw3-data-release/train/")
    total_size = len(complete_dataset)
    segment_size = total_size // 10

    for epoch in range(150):
        cumulative_loss = 0
        batch_count = 0

        for segment in range(10):
            # Define segment boundaries
            seg_start = segment * segment_size
            seg_end = seg_start + segment_size if segment < 9 else total_size
            segment_range = list(range(seg_start, seg_end))
            segment_data = torch.utils.data.Subset(complete_dataset, segment_range)
            data_iterator = DataLoader(segment_data, batch_size=1, shuffle=True, 
                                     collate_fn=lambda x: tuple(zip(*x)))

            print(f"Epoch {epoch + 1}, processing segment {segment + 1}")

            grad_accumulation = 8
            training_optimizer.zero_grad()

            for batch_num, (input_images, annotations) in enumerate(tqdm(data_iterator)):
                input_images = [img.to(computing_device) for img in input_images]
                annotations = [{k: v.to(computing_device) for k, v in t.items()} for t in annotations]

                with torch.amp.autocast(device_type='cuda'):
                    loss_components = trained_model(input_images, annotations)
                    total_loss = sum(loss for loss in loss_components.values()) / grad_accumulation

                precision_scaler.scale(total_loss).backward()

                if (batch_num + 1) % grad_accumulation == 0 or (batch_num + 1) == len(data_iterator):
                    precision_scaler.step(training_optimizer)
                    precision_scaler.update()
                    training_optimizer.zero_grad()

                cumulative_loss += total_loss.item() * grad_accumulation
                batch_count += 1

                del input_images, annotations, loss_components, total_loss
                torch.cuda.empty_cache()
                gc.collect()

            time.sleep(5)

        
        mean_loss = cumulative_loss / batch_count
        print(f"[Epoch {epoch + 1}] Mean Loss across segments: {mean_loss:.4f}")
        epoch_losses.append(mean_loss)

        if mean_loss < lowest_loss:
            lowest_loss = mean_loss
            stop_patience = 0
            torch.save(trained_model.state_dict(), f"optimal_model_epoch{epoch + 1:03d}_loss{mean_loss:.4f}.pth")
            print(f"Updated best model at epoch {epoch + 1}")
        else:
            stop_patience += 1
            print(f"No progress, patience counter: {stop_patience}")
            if stop_patience >= max_patience:
                break

        graph_file = f"loss_graph_epoch{epoch + 1:03d}.png"
        plot_loss(epoch_losses, graph_file)
        print(f"Loss graph saved to {graph_file}")


def plot_loss(loss_history, save_path):
    """
    Plot and save the training loss graph.
    """
    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


# Start the training process
if __name__ == "__main__":
    train_model()