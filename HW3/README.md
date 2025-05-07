# NYCU-Computer-Vision-2025-Spring-HW3
**StudentID**: 109612019  
**Name**: Albert Lin 林伯偉

## Introduction
This repository contains the code for the second homework assignment of the NYCU Computer Vision 2025 Spring course. This assignment implements an instance segmentation model to detect and classify four types of cells in medical images. The model utilizes a ResNetRS200 backbone "timm/resnetrs200.tf_in1k", trained on a dataset of cell images, with the goal of achieving high performance in segmentation and classification tasks. The training process incorporates data augmentation techniques to improve the model's generalization capability.

The model's performance is evaluated based on both training loss and private scoring on Codabench, Additionally, to manage memory usage, each epoch during training is split into 10 segments, with only 10% of the dataset loaded at a single segment.

## How to install
  To run the code in this repository, follow these steps:  

#### 1. Clone this repository:

#### 2. Create a conda environment:  
    conda create -n <env_name> python=3.9

#### 3. Download the dataset  
    https://drive.google.com/file/d/1B0qWNzQZQmfQP7x7o4FDdgb9GvPDoFzI/view

#### 4. To run the training process, execute the following command:  
    python train.py

#### 5.To generate the .json predictions file and the segmentation results, run the following command:  
    python predict.py  

## Performance snapshot  
![loss_plot-resnetrs200-dataaug-round121](https://github.com/user-attachments/assets/059bdd5b-fee3-46f8-8139-2041c9466bd3)
![Screenshot 2025-05-08 at 12 00 00 AM](https://github.com/user-attachments/assets/fb94d74b-8929-42d8-a9a2-e4ee8e210102)

## Instance Segmentation results
![result_image_96](https://github.com/user-attachments/assets/dd9a9aff-7b58-4752-a078-dafa7f94564e)
![result_image_3](https://github.com/user-attachments/assets/afa5bdaf-9861-42d0-83f4-490011893ebc)
![result_image_6](https://github.com/user-attachments/assets/14b563ce-a357-4094-a9c4-4c23e030e937)
