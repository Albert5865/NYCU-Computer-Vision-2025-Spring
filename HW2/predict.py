import torch
import torchvision.models.detection as models
from dataset import DigitDataset
from torch.utils.data import DataLoader
import json
import os
import gc
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.v2 as transforms
from utils import write_pred_csv


def predict(model, test_dir):
    results = []
    digit_map = {}

    for fname in os.listdir(test_dir):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

        img_id = int(fname.replace(".png", ""))
        img = Image.open(os.path.join(test_dir, fname)).convert("RGB")
        img_tensor = transforms.ToTensor()(img).to(device)
        with torch.no_grad():
            outputs = model([img_tensor])[0]

        boxes = outputs["boxes"].cpu()
        scores = outputs["scores"].cpu()
        labels = outputs["labels"].cpu()

        digits = []
        for box, score, label in zip(boxes, scores, labels):
            if score < 0.65:
                continue
            results.append({
                "image_id": img_id,
                "bbox": [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])],
                "score": float(score),
                "category_id": int(label)
            })
            digits.append((box[0], str(label.item()-1)))  # Convert to string and subtract 1 for zero-based index

        if digits:
            digits = sorted(digits, key=lambda x: x[0])  # sort by x position
            pred_label = ''.join(d[1] for d in digits)
        else:
            pred_label = -1

        digit_map[img_id] = pred_label

    with open("output/pred.json", "w") as f:
        json.dump(results, f, indent=4)

    write_pred_csv("output/pred.csv", digit_map)

if __name__ == "__main__":
    model = models.fasterrcnn_resnet50_fpn_v2(weights=None)
    model.load_state_dict(torch.load("best_model.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predict(model, "nycu-hw2-data/test")
