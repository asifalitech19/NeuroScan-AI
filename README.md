# NeuroScan-AI 🔍🏭

**Industrial Surface Defect Segmentation and Anomaly Detection using Deep Residual U-Net (ResUNet) and Convolutional Autoencoders**

This repository contains an end-to-end industrial computer vision inspection system developed to detect and segment unseen surface defects on manufacturing materials. It replaces subjective manual inspection with high-precision, pixel-level deep learning architectures[cite: 2].

This project was developed as a comprehensive open-ended laboratory implementation for the Artificial Intelligence program at Iqra University.

## 📌 Project Architecture
The pipeline executes a dual-model approach to handle complex, non-linear texture anomalies:
1. **Anomaly Detection:** An Unsupervised Convolutional Autoencoder that compresses defect-free images into a latent space and flags defects based on high Mean Squared Error (MSE) reconstruction loss[cite: 2].
2. **Semantic Segmentation:** A Deep Residual U-Net (ResUNet) utilizing ResNet bottleneck blocks to perform fine-grained, pixel-level masking of defective regions[cite: 2].

## 📂 Repository Structure
```text
NeuroScan-AI/
├── data/
│   └── dataset_loader.py             # Custom PyTorch Dataset, XML bounding box to pseudo-mask conversion
├── models/
│   ├── ResUNet.py                    # Deep ResUNet with skip connections and residual blocks
│   ├── Autoencoder.py                # Unsupervised Convolutional Autoencoder architecture
│   └── StandardCNN.py                # Baseline CNN Encoder-Decoder
├── notebooks/
│   └── Training_and_Evaluation.ipynb # Complete model training, evaluation, and visual plotting
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation and performance metrics


## 📊 Dataset & Uniqueness Mechanism
* **Dataset:** NEU Metal Surface Defect Database (1,800+ images across 6 defect classes)[cite: 2].
* **Uniqueness:** A unique random seed **19** (derived from student Roll Number parameters) was enforced during data splitting and augmentation pipelines to ensure fully reproducible and unique experimental outputs.

## 🚀 Performance Metrics
Evaluated on severe low-contrast defects and noisy conditions. Inference speed tested on NVIDIA Tesla T4 GPU.

| Model Architecture | Validation IoU | Dice Score | Precision | Inference Speed |
| :--- | :--- | :--- | :--- | :--- |
| **Standard CNN Encoder-Decoder** | 0.5319 | 0.6743 | 0.6951 | ~130.5 FPS |
| **Convolutional Autoencoder** | N/A | N/A | (MSE Loss: 0.0220) | ~160.0 FPS |
| **Deep ResUNet** | **0.6737** | **0.7907** | **0.8656** | **155.42 FPS** |

## 🛠️ Usage
Clone the repository:
```bash
git clone [https://github.com/asifalitech19/NeuroScan-AI.git](https://github.com/YourUsername/NeuroScan-AI.git)
