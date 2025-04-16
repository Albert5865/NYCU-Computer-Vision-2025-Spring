# NYCU-Computer-Vision-2025-Spring-HW2
**StudentID**: 109612019  
**Name**: Albert Lin 林伯偉

## Introduction
In this assignment, I implemented Faster R-CNN (Region-based Convolutional Neural Networks) with a ResNet-50 backbone to detect and classify digits in images. The model was trained using the provided dataset of handwritten digits, and I evaluated its performance using mean Average Precision (mAP). The training process also involved experimenting with learning rate schedulers (CosineAnnealingLR) to improve convergence.The model was evaluated on both training loss and validation mAP. Additionally, I used random sampling to manage memory usage, as only 10% of the dataset was trained on per epoch.The project uses the **PyTorch** framework and **TensorBoard** for visualization of training and validation metrics.

## How to install
  To run the code in this repository, follow these steps:  

#### 1. Clone the repository:
    git clone https://github.com/Albert5865/NYCU-Computer-Vision-2025-Spring-HW1.git  

#### 2. Create a conda environment:  
    conda create -n <env_name> python=3.9

#### 3. Download the dataset  
    https://drive.google.com/file/d/1fx4Z6xl5b6r4UFkBrn5l0oPEIagZxQ5u/view  

#### 4. To run the training process, execute the following command:  
    python train.py

#### 5.To generate predictions, run the following command:  
    python predict.py  

## Performance snapshot  
    Best Validation Accuracy: 70.28%  
    Best Validation Loss: 85.67%  
    Final Test Accuracy (Public): 91%




