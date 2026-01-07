# QualiSight-AI: Industrial Anomaly Detection via Segmentation

This project implements a deep learning solution for detecting anomalies in industrial images using semantic segmentation. It utilizes a U-Net architecture trained on the MVTec Anomaly Detection dataset using PyTorch, and includes an evaluation pipeline using ONNX Runtime.

## Project Overview

The goal of this project is to identify defective regions in images of manufactured goods. Instead of simple classification (defective vs. good), this model performs segmentation to output a pixel-wise mask indicating the exact location of defects.

### Key Features
* **Model Architecture:** Uses a custom U-Net implementation for segmentation tasks.
* **Loss Function:** Combines Binary Cross-Entropy (BCE) and Dice Loss for robust training on imbalanced segmentation tasks.
* **Data Augmentation:** Utilizes the `albumentations` library for augmentations like resizing, flipping, rotation, and normalization.
* **Training Monitoring:** Saves visual results (original, ground truth, heatmap overlay) periodically during training to the `results_visuals` directory.
* **Model Export:** Automatically exports the best-performing model (lowest loss) to ONNX format for efficient inference.
* **Evaluation Pipeline:** Includes a dedicated script (`eval.py`) using ONNX Runtime to calculate Dice scores and classification accuracy on test sets.

---

## Visual Results

The training process automatically generates visualizations comparing the input image, the ground truth mask, and the model's predicted heatmap.

### Training Progress Example
Below is an example of the model's output during training. It shows the original image, the actual defect mask (Ground Truth), and the model's prediction overlaid as a heatmap.

![Training Progress Example](./results_visuals/YOUR_IMAGE_HERE.png)
*(Figure: Left: Original Image, Center: Ground Truth Defect Mask, Right: Predicted Heatmap Overlay)*

---

## Getting Started

### Prerequisites

Ensure you have Python installed (tested with Python 3.x). Install the required dependencies:

```bash
pip install torch torchvision torchaudio opencv-python numpy matplotlib tqdm albumentations onnxruntime-gpu
