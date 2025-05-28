import os
import numpy as np
from PIL import Image
import glob

# Placeholder for PromptIR imports (modify based on actual PromptIR code)
# from promptir import PromptIRModel, train_model, test_model

# Define dataset paths (adjust based on your local paths)
DATASET_PATHS = {
    'derain': (
        'data/train/degraded/rain-*.png',
        'data/train/clean/rain_clean-*.png'
    ),
    'desnow': (
        'data/train/degraded/snow-*.png',
        'data/train/clean/snow_clean-*.png'
    )
}

TEST_PATH = 'test/degraded/'
OUTPUT_DIR = 'output/test/'
OUTPUT_NPZ = 'pred.npz'

# Step 1: Modify PromptIR configuration (example)
def configure_promptir():
    """
    Modify PromptIR to include snow as a degradation type.
    This is a placeholder; actual implementation depends on PromptIR's code.
    """
    degradation_types = ['derain', 'desnow']
    # Update PromptIR's data.py or config.py to include:
    # dataset_paths = DATASET_PATHS
    # degradation_types = ['derain', 'desnow']
    print("Configured PromptIR with degradation types:", degradation_types)

# Step 2: Train the model (example command)
def train_model():
    """
    Train PromptIR on rain and snow datasets.
    Run: python train.py --de_type derain desnow
    """
    os.system('python train.py --de_type derain desnow')
    print("Training completed.")

# Step 3: Test the model (example command)
# def test_model():
#     """
#     Test PromptIR on the test set using all-in-one mode.
#     Run: python test.py --mode 3 --test_path test/degraded/
#     """
#     if not os.path.exists(OUTPUT_DIR):
#         os.makedirs(OUTPUT_DIR)
#     os.system(f'python test.py --mode 3 --test_path {TEST_PATH}')
#     print(f"Restored images saved to {OUTPUT_DIR}")

def test_model():
    """
    Test PromptIR on the test set using all-in-one mode.
    Run: python test.py --mode 3 --derain_path test/degraded/ --output_path output/test/
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    os.system(f'python test.py --mode 3 --derain_path {TEST_PATH} --output_path {OUTPUT_DIR}')
    print(f"Restored images saved to {OUTPUT_DIR}")


# Step 4: Convert restored images to pred.npz
def save_to_npz():
    """
    Convert restored images to pred.npz format.
    Modified from example_img2npz.py.
    """
    images_dict = {}
    for img_path in glob.glob(os.path.join(OUTPUT_DIR, '*.png')):
        filename = os.path.basename(img_path)
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img).transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        images_dict[filename] = img_array
    np.savez(OUTPUT_NPZ, **images_dict)
    print(f"Saved {len(images_dict)} images to {OUTPUT_NPZ}")

# Main execution
if __name__ == "__main__":
    configure_promptir()
    train_model()
    test_model()
    save_to_npz()