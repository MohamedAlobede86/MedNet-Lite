# MedNet-Lite 🧠⚡  
**Interpretable Lightweight CNN for Emergency Biomedical Signal Classification**

MedNet-Lite is a compact and interpretable convolutional neural network designed for real-time classification of ultra-short biomedical signals in emergency care. It integrates Convolutional Block Attention Modules (CBAM) to enhance feature focus and provide visual interpretability, making it suitable for deployment on resource-constrained, 5G-enabled edge devices.

---

## 📌 Objectives
- Enable rapid and reliable triage decisions using short, single-channel biomedical signals.
- Minimize false negatives to prioritize clinical safety.
- Provide built-in interpretability through attention maps.
- Ensure compatibility with low-power, real-time systems in emergency environments.

---

## 🧪 Dataset

The dataset used in this study was sourced from the [Waveform Biomedical Signals Repository on Kaggle](https://www.kaggle.com/datasets/your-dataset-link).  
It contains 5,000 samples, each with 5 time points across a single physiological channel.  
Binary labels were assigned using a threshold-based criterion:  
- **Class 1 (Critical)**: mean normalized value > 0.5  
- **Class 0 (Non-Critical)**: otherwise  

This simulates emergency triage logic where rapid decisions are based on minimal signal input.