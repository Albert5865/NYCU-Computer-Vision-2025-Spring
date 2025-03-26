import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10
NUM_CLASSES = 100
MODEL_PATH = "model.pth"

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load datasets
train_dataset = ImageFolder(root="./data/train", transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = ImageFolder(root="./data/val", transform=transform)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = timm.create_model(
    "seresnext101d_32x8d.ah_in1k",
    pretrained=True,
    num_classes=NUM_CLASSES,
    drop_rate=0.5
)
model = model.to(device)

model_previous = torch.load("model-epoch8-val-loss 1.05462164-acc 75.33%.pth")
model.load_state_dict(model_previous)
del model_previous

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.65, patience=0)

# Track performance
train_losses = []
val_losses = []
val_accuracies = []
training_accuracies = []
best_accuracy = 0.0
best_val_loss = float('inf')

writer = SummaryWriter(log_dir='runs/seresnext101_experiment')

# Training loop
total_batches = 0
total_val_batches = 0

for epoch in range(1, NUM_EPOCHS + 1):
    torch.mps.empty_cache()
    gc.collect()
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        enumerate(train_loader, start=1),
        total=len(train_loader),
        desc=f"Epoch {epoch}/{NUM_EPOCHS} (LR: {optimizer.param_groups[0]['lr']:.6f})"
    )

    for _, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()
        total_batches += 1
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = running_loss / len(train_loader)
    training_accuracy = 100 * correct / total

    train_losses.append(avg_train_loss)
    training_accuracies.append(training_accuracy)

    # Validation
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)

            loss = criterion(val_outputs, val_labels)
            running_val_loss += loss.item()

            _, predicted = torch.max(val_outputs, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
            total_val_batches += 1

    avg_val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    scheduler.step(avg_val_loss)

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    writer.add_scalar("Loss/Train", avg_train_loss, epoch)
    writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/Train", training_accuracy, epoch)
    writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

    print(f"Epoch [{epoch}/{NUM_EPOCHS}], Training Loss: {avg_train_loss:.4f}")
    print(f"Epoch [{epoch}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}")
    print(f"Training Accuracy: {training_accuracy:.2f}%")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    if avg_val_loss < best_val_loss or val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_val_loss = avg_val_loss
        torch.save(
            model.state_dict(),
            f"model-epoch{epoch}-val-loss{best_val_loss:.8f}-acc{val_accuracy:.2f}%.pth"
        )
        print(f"New best model saved with val loss: {best_val_loss:.2f} accuracy: {best_accuracy:.2f}%")

writer.close()

# Plot: Loss over epochs
plt.figure()
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.xticks(range(1, len(train_losses) + 1))
plt.grid(True)
plt.savefig("training_loss.png")
plt.close()

# Plot: Accuracy over epochs
plt.figure()
plt.plot(range(1, len(training_accuracies) + 1), training_accuracies, label='Training Accuracy')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.xticks(range(1, len(training_accuracies) + 1))
plt.grid(True)
plt.savefig("training_accuracy.png")
plt.close()
