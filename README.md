# Chili Plant Disease Classification using VGG16 & Fine-Tuning

## Project Overview
This mini deep learning project focuses on **classifying chili plant diseases** using **Convolutional Neural Networks (CNN)** and **Transfer Learning** with **VGG16** (a pre-trained model on ImageNet).  
Two phases of training were implemented:

1. **Base Training (Feature Extraction)** — using VGG16 as a fixed feature extractor.  
2. **Fine-Tuning** — unfreezing part of VGG16 to adapt deeper layers to our specific dataset.

---

## Motivation
Training a deep CNN from scratch requires a large dataset and high computational power.  
Using **Transfer Learning** allows us to:
- Reuse the powerful feature extraction capability of **VGG16**.
- Train faster with fewer data.
- Improve generalization and performance on specific tasks (like plant disease detection).

---

## Architecture

### Base Model: VGG16
- Pre-trained on **ImageNet**.
- Loaded **without the top fully-connected layers** (`include_top=False`).
- Input shape: **(224, 224, 3)**.

### Added Custom Layers
| Layer | Description |
|-------|--------------|
| `GlobalAveragePooling2D` | Converts feature maps into a 1D vector. |
| `Dense(256, activation='relu')` | Fully connected layer for learning complex patterns. |
| `Dropout(0.5)` | Reduces overfitting by randomly disabling neurons. |
| `Dense(num_classes, activation='softmax')` | Outputs class probabilities. |

### Phase 1: Base Training
- VGG16 layers are **frozen** (`trainable=False`).
- Only custom layers are trained.
- Optimizer: **Adam (lr=0.0001)**.

### Phase 2: Fine-Tuning
- Last convolutional blocks of VGG16 are **unfrozen**.
- Model recompiled with **lower learning rate (1e-5)**.
- Both the pre-trained and custom layers learn together for better generalization.

---

## Results Summary

| Metric | Base Training | Fine-Tuning |
|---------|----------------|-------------|
| **Training Loss** | Decreases regularly | Decreases faster and stabilizes lower |
| **Validation Loss** | Stable but slightly higher | Lower and more stable |
| **Training Accuracy** | Gradual improvement | Faster convergence |
| **Validation Accuracy** | Fluctuating | More consistent and higher overall |
| **Overfitting** | Slight | Reduced |
| **Interpretation** | Model learns basic patterns | Model learns deeper, more robust features |

**Fine-tuning improved overall accuracy and generalization.**

---

## Technologies & Libraries
- **Python 3.10+**
- **TensorFlow / Keras**
- **NumPy**
- **Pickle**
- **Matplotlib (for visualization)**
- **ImageDataGenerator (for data augmentation)**

---

## 📁 Project Structure
