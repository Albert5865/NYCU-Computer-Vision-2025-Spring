# NYCU-Computer-Vision-2025-Spring-HW1
**StudentID**: 109612019  
**Name**: Albert Lin 林伯偉

## Introduction
This repository contains the code for the first homework assignment of the **NYCU Computer Vision 2025 Spring** course. The task focuses on training and evaluating an image classification model using various deep learning techniques, including regularization, data augmentation, and learning rate scheduling. The assignment involves applying a deep neural network architecture (SEResNeXt101d_32x8d.ah_in1k) for classifying a dataset of 100 object categories.

The model is trained using standard techniques such as:
- **Dropout**: Applied before the final classification layer to prevent overfitting.
- **Data Augmentation**: Includes Gaussian blur, random rotation, and random horizontal flipping to improve generalization.
- **Weight Decay (L2 Regularization)**: Applied to avoid overfitting and to ensure that the model learns simpler, more generalizable patterns.
- **Dynamic Learning Rate Scheduling**: Utilized through the `ReduceLROnPlateau` scheduler to adjust the learning rate when validation performance plateaus.

The project uses the **PyTorch** framework and **TensorBoard** for visualization of training and validation metrics.

## How to install
To run the code in this repository, follow these steps:  

### 1. Clone the repository:
git clone https://github.com/your_username/NYCU-Computer-Vision-2025-Spring-HW1.git  

### 2. Create a conda environment:  

### 3. Download the dataset  
https://drive.google.com/file/d/1fx4Z6xl5b6r4UFkBrn5l0oPEIagZxQ5u/view  

### 4. To run the training process, execute the following command:  
python train.py

### 5.To generate predictions, run the following command:  
python predict.py  

## Performance snapshot  
Best Validation Accuracy: 70.28%  
Best Validation Loss: 85.67%  
Final Test Accuracy (Public): 91%




