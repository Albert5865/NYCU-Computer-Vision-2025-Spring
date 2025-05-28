# NYCU-Computer-Vision-2025-Spring-HW4
**StudentID**: 109612019  
**Name**: Albert Lin 林伯偉

## Introduction
This repository contains the code for the fourth homework assignment of the NYCU Computer Vision 2025 Spring course. This assignment implements an image restoration model to remove rain and snow degradations from images using the PromptIR framework. The model is trained on a dataset of 1600 degraded images per type (rain and snow) with corresponding clean images, aiming to achieve high performance on a test set of 50 images per type. The training process incorporates data augmentation (RandomRotation, ColorJitter), a modified loss function (0.7 * L1 + 0.3 * SSIM), and an adaptive learning rate scheduler (ReduceLROnPlateau) to enhance generalization and stability.

The model's performance is evaluated using Peak Signal-to-Noise Ratio (PSNR) on both public and private leaderboards via Codabench. To manage memory usage, the batch size is set to 2, and the training process leverages PyTorch Lightning with CUDA acceleration.

## How to install
  To run the code in this repository, follow these steps:  

#### 1. Clone this repository:

#### 2. Create a conda environment:  
    conda create -n <env_name> python=3.9

#### 3. Download the dataset  
    https://drive.google.com/drive/folders/1Q4qLPMCKdjn-iGgXV_8wujDmvDpSI1ul

#### 4. Rename the dataset folder as "data" 

    
#### 4. To run the training process, execute the following command:  
    python main.py

#### 5. The running main.py will automatically start testing and generate the "pred.npz" file after training is finished.


## Performance snapshot  
![Screenshot 2025-05-28 at 9 06 44 PM](https://github.com/user-attachments/assets/a93afa9b-7a6b-4043-ba70-cf87a6ea9f85)


## Denoise results

![4](https://github.com/user-attachments/assets/c5265a49-2a50-4a71-ae82-2fbc27f554fe)![4](https://github.com/user-attachments/assets/14a466e8-82b6-4054-85e0-c39916b5984a)

![59](https://github.com/user-attachments/assets/9f5ec103-a748-405f-80c6-c918446802c3)![59](https://github.com/user-attachments/assets/d2e7b466-cdca-4b10-b750-827ee7dd4e73)

![91](https://github.com/user-attachments/assets/5dfc95e2-a75a-46b7-84fb-eb192b8b7a09)![91](https://github.com/user-attachments/assets/50300db3-4ce9-4ed7-b6e4-6a610c91acc4)

![93](https://github.com/user-attachments/assets/5c0efbe5-3f8b-430d-af7d-a993762c9c31)![93](https://github.com/user-attachments/assets/1f3d55b3-264c-4c0a-b53d-e260891aea47)
