import os
import numpy as np
from PIL import Image
import glob

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

TEST_PATH = 'data/test/degraded/'
OUTPUT_DIR = 'output/test/'
OUTPUT_NPZ = 'pred.npz'

def configure_promptir():
    degradation_types = ['derain', 'desnow']
    print("Configured PromptIR with degradation types:", degradation_types)

def train_model():
    result = os.system('python train.py --de_type derain desnow --mps 1')
    if result == 0:
        print("Training completed.")
    else:
        print("Training failed.")
        exit(1)

def test_model():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    result = os.system(f'python test.py --mode 3 --test_path {TEST_PATH} --output_path {OUTPUT_DIR} --mps 1')
    if result == 0:
        print(f"Restored images saved to {OUTPUT_DIR}")
    else:
        print("Testing failed.")
        exit(1)

def save_to_npz():
    images_dict = {}
    for img_path in glob.glob(os.path.join(OUTPUT_DIR, '*.png')):
        filename = os.path.basename(img_path)
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img).transpose(2, 0, 1)
        images_dict[filename] = img_array
    if not images_dict:
        print("No images found in output directory. NPZ file not created.")
        exit(1)
    np.savez(OUTPUT_NPZ, **images_dict)
    print(f"Saved {len(images_dict)} images to {OUTPUT_NPZ}")

if __name__ == "__main__":
    # configure_promptir()
    # train_model()
    test_model()
    save_to_npz()