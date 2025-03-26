import os
import torch
import pandas as pd
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


# Model parameters
NUM_CLASSES = 100
MODEL_PATH = "model-epoch6-val-loss 0.80919652.pth"

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = ImageFolder(root="./data/train", transform=transform)


class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.image_filenames = sorted(os.listdir(test_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.test_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.splitext(self.image_filenames[idx])[0]


# Load test dataset
test_dataset = TestDataset(test_dir="./data/test", transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model = timm.create_model("seresnext101d_32x8d.ah_in1k")
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)
model.eval()

# Predict and store results
predictions = []
image_filenames = []

with torch.no_grad():
    for images, filenames in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for fn, pred in zip(filenames, predicted.cpu().numpy()):
            pred_label = train_dataset.classes[pred]
            predictions.append(pred_label)
            image_filenames.append(fn)

# Save predictions to CSV
prediction_df = pd.DataFrame({
    "image_name": image_filenames,
    "pred_label": predictions
})
prediction_df.to_csv("prediction.csv", index=False)

print("Predictions saved to prediction.csv")
