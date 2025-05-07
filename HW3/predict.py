import torch
import json
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools import mask as mask_utils
from train import create_model  # Import the create_model function from train.py
from PIL import Image
import tifffile
import matplotlib.pyplot as plt
import cv2  # For image resizing

# Define a color map for different categories
COLORS = {
    1: (0, 255, 0),    # Green for category 1
    2: (255, 0, 0),    # Red for category 2
    3: (0, 0, 255),    # Blue for category 3
    4: (255, 255, 0)   # Yellow for category 4
}

# Define the Dataset for Test Images
class TestDataset(Dataset):
    def __init__(self, images_dir, json_file):
        self.images_dir = images_dir
        with open(json_file, 'r') as f:
            self.image_info = json.load(f)
        
        self.image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.tif')])
        self.image_ids = {item['file_name']: item['id'] for item in self.image_info}
        self.image_details = {item['file_name']: {'id': item['id'], 'height': item['height'], 'width': item['width']} for item in self.image_info}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_name = os.path.basename(img_path)
        image_details = self.image_details.get(image_name, None)

        if image_details is None:
            raise ValueError(f"Image {image_name} not found in the test_image_name_to_ids.json file")

        image_id = image_details['id']
        image = tifffile.imread(img_path)
        image = Image.fromarray(image).convert("RGB")  # Convert to RGB
        image_tensor = transforms.ToTensor()(image)  # Convert to tensor

        # Return image along with its corresponding image details
        return image_tensor, {"image_id": image_id, "height": image_details['height'], "width": image_details['width']}

# Function to visualize instance segmentation results and save to a folder
def visualize_instance_segmentation(image, boxes, masks, labels, scores, image_id, output_dir="segmentation_results"):
    """
    Visualizes the instance segmentation results by filling the cell shapes with unique colors for each category
    and saves the result to a specified folder.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(1, figsize=(12, 9))
    # Convert image tensor to NumPy array for display (from [C, H, W] to [H, W, C])
    image = image.cpu().numpy().transpose(1, 2, 0) * 255  # Undo normalization and transpose
    image = image.astype(np.uint8)

    # Create a copy of the image to draw filled masks
    overlay = image.copy()

    for i in range(len(boxes)):
        box = boxes[i]
        mask = masks[i]
        label = labels[i]
        score = scores[i]

        # Get color for the current category
        color = COLORS.get(label, (255, 255, 255))  # Default to white if label not in COLORS

        # Resize mask to match the original image dimensions
        mask_binary = mask[0] > 0.5  # Assuming binary mask
        mask_resized = cv2.resize(mask_binary.astype(np.uint8), (image.shape[1], image.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)

        # Fill the mask with the category color
        overlay[mask_resized.astype(bool)] = color  # Fill the mask area with the category color

        # Add label and score text
        # ax.text(box[0], box[1], f'{label}: {score:.2f}', color='black', fontsize=12, 
        #         bbox=dict(facecolor=[c / 255 for c in color], alpha=0.7))

    # Display the overlaid image and save it
    ax.imshow(overlay)
    output_path = os.path.join(output_dir, f"result_image_{image_id}.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

# Function to generate the JSON file and save visualization results
def generate_submission_json(model, test_loader, device, save_path="test-result.json", output_dir="segmentation_results"):
    model.eval()
    results = []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Processing test set"):
            images = [image.to(device) for image in images]
            output = model(images)

            for idx, result in enumerate(output):
                image_id = targets[idx]['image_id']
                boxes = result['boxes'].cpu().numpy()
                labels = result['labels'].cpu().numpy()
                masks = result['masks'].cpu().numpy()
                scores = result['scores'].cpu().numpy()

                # Convert masks to RLE format for JSON
                rles = []
                for mask in masks:
                    binary_mask = (mask[0] > 0.5).astype(np.uint8)
                    rle = mask_utils.encode(np.asfortranarray(binary_mask))
                    rle['counts'] = rle['counts'].decode('ascii')  # Fix the bytes issue
                    rles.append(rle)

                # Prepare each image result for JSON
                for i in range(len(boxes)):
                    result_data = {
                        "image_id": int(image_id),
                        "score": scores[i].item(),
                        "category_id": int(labels[i]),
                        "segmentation": rles[i],  # RLE dictionary
                        "bbox": boxes[i].tolist(),
                    }
                    results.append(result_data)

                # Save the visualization result using the correct image from the batch
                visualize_instance_segmentation(images[idx], boxes, masks, labels, scores, image_id, output_dir)

    # Save results to JSON file
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Saved submission file to {save_path}")
    print(f"Segmentation result images saved to {output_dir}")

# Initialize model and dataset for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5  # Adjust to the number of classes

# Use create_model from train.py to load the model
model = create_model(num_classes=num_classes).to(device)

# Load the pre-trained model (update with your model's file path)
model.load_state_dict(torch.load('resnetrs200-dataaug_round047_loss0.8602-0.29.pth'))

# Load the test dataset (images only, no masks)
test_dataset = TestDataset(
    images_dir='hw3-data-release/test_release',  # Update with the path to your test folder
    json_file='hw3-data-release/test_image_name_to_ids.json'  # Path to the JSON file
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Generate the submission JSON and save visualization results
generate_submission_json(model, test_loader, device, save_path="test-results.json")