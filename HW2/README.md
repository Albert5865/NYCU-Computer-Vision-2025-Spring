# NYCU-Computer-Vision-2025-Spring-HW2
**StudentID**: 109612019  
**Name**: Albert Lin 林伯偉

## Introduction
This repository contains the code for the second homework assignment of the NYCU Computer Vision 2025 Spring course. This assignment implements Faster R-CNN (Region-based Convolutional Neural Networks) with a ResNet-50 backbone to detect and classify digits in images. The model is trained using a provided dataset of handwritten digits, and its performance is evaluated using mean Average Precision (mAP). The training process includes the use of learning rate schedulers, specifically CosineAnnealingLR, to improve convergence.

The model's performance is evaluated in terms of both training loss and validation mAP. Additionally, random sampling is employed to manage memory usage by training on only 10% of the dataset per epoch.The project uses the **PyTorch** framework and **TensorBoard** for visualization of training and validation metrics.

## How to install
  To run the code in this repository, follow these steps:  

#### 1. Clone the repository:
    https://github.com/Albert5865/NYCU-Computer-Vision-2025-Spring.git

#### 2. Create a conda environment:  
    conda create -n <env_name> python=3.9

#### 3. Download the dataset  
    https://drive.google.com/file/d/13JXJ_hIdcloC63sS-vF3wFQLsUP1sMz5/view

#### 4. To run the training process, execute the following command:  
    python train.py

#### 5.To generate predictions, run the following command:  
    python predict.py  

## Performance snapshot  
  ![map_curve4](https://github.com/user-attachments/assets/bfcd3a2f-e85f-4a1f-8e01-7485a3ade9da)
  ![score1-cosine](https://github.com/user-attachments/assets/8dee024d-bbe4-4cb1-98bb-25b68372ad99)
  ![score2-cosine](https://github.com/user-attachments/assets/9afbfffb-61fd-435c-b907-f47ee4fa7d99)



