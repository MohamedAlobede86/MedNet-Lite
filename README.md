# MedNet-Lite

MedNet-Lite is a lightweight and interpretable 1D-CNN architecture for real-time ECG anomaly screening on 5G-enabled edge devices. It integrates a 1D-adapted Convolutional Block Attention Module (CBAM) to provide physiologically grounded interpretability, making it suitable for non-clinical health monitoring systems such as wearables and remote sensing platforms.

---

## 📊 Key Features

- ✅ Compact 1D-CNN architecture with only 18,021 parameters
- ✅ Built-in interpretability via temporal attention maps (CBAM)
- ✅ Ultra-low inference latency: **1.10 ms**
- ✅ Tiny model size: **79 KB**
- ✅ Designed for edge deployment (Jetson Nano, Raspberry Pi)
- ✅ Evaluated on real-world ECG data (MIT-BIH)

---

## 📁 Dataset

This project uses the publicly available **MIT-BIH Arrhythmia Database**:

🔗 [https://physionet.org/content/mitdb/1.0.0/](https://physionet.org/content/mitdb/1.0.0/)

- 112,551 ECG beats extracted from Lead II
- Sampling rate: 360 Hz
- Binary classification: Normal vs Abnormal beats

---

## 🧠 Model Architecture

MedNet-Lite consists of:

- Two convolutional blocks with BatchNorm and ReLU
- Two 1D-adapted CBAM modules (channel + temporal attention)
- Adaptive pooling and fully connected output layer
- Sigmoid activation for binary classification

Attention maps consistently highlight the QRS complex region (~sample 142), enhancing physiological interpretability.

---

## ⚙️ How to Run

### 1. Preprocess the dataset

```bash
python preprocess.py
