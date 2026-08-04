# MedNet-Lite: An Interpretable Ultra-Lightweight 1D-CNN with CBAM for Real-Time ECG Anomaly Screening

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official implementation of the revised manuscript: **"MedNet-Lite: An Interpretable Ultra-Lightweight 1D-CNN with CBAM for Real-Time ECG Anomaly Screening in Resource-Constrained Edge Computing Environments"** (Manuscript ID: BSPC-D-25-13749).

## 📌 Abstract
Deploying artificial intelligence (AI) in Internet of Things (IoT) healthcare ecosystems requires models that balance predictive reliability, computational efficiency, and intrinsic interpretability under strict edge computing constraints. This repository contains the complete, reproducible code for **MedNet-Lite**, an interpretable, ultra-lightweight 1D convolutional neural network (1D-CNN) incorporating a 1D-adapted Convolutional Block Attention Module (CBAM) for real-time electrocardiogram (ECG) anomaly screening.

## 📊 Key Features & Final Results
- ⚡ **Ultra-Lightweight:** Only **4,222** trainable parameters and **0.0249 MB** model size.
- 🎯 **High Performance:** Achieves **98.40%** accuracy and **0.942** PR-AUC on the MIT-BIH Arrhythmia Database.
- 🔍 **Intrinsic Explainability:** Embedded 1D-CBAM provides attention maps without post-hoc computational overhead (Weighted attention centroid RMSE: **5.84 samples**; QRS energy concentration: **14.21%**).
- 🚀 **Edge-Ready:** Optimized for deployment with ONNX Runtime, achieving **0.34 ms** inference latency and **4.82 MB** peak RAM consumption (End-to-End latency: **2.28 ms**).
- 🛡️ **Robustness:** Validated via rigorous 5-Fold Stratified Cross-Validation and formal Ablation Studies proving CBAM's critical role in minority-class detection.

## 📁 Dataset
This project uses the publicly available **MIT-BIH Arrhythmia Database**:  
🔗 [https://physionet.org/content/mitdb/1.0.0/](https://physionet.org/content/mitdb/1.0.0/)

- **Final Processed Size:** 101,406 ECG beats (rigorously filtered to eliminate boundary artifacts).
- **Sampling rate:** 360 Hz.
- **Classification:** Binary classification (Normal vs. Abnormal) based on AAMI EC57 recommendations.

## 🧠 Model Architecture
MedNet-Lite consists of:
1. Two 1D convolutional blocks (16 channels/k15, 32 channels/k7) with BatchNorm, ReLU, and Max Pooling.
2. One integrated 1D-adapted CBAM module (channel + spatial attention) for physiologically grounded interpretability.
3. Adaptive global average pooling and a fully connected output layer.

*Note: Attention maps consistently highlight the QRS complex region, enhancing physiological interpretability without adding significant computational overhead.*

## 📂 Repository Structure
The complete end-to-end pipeline is organized into three sequential Jupyter Notebooks for maximum reproducibility:

1. `01_Data_Preprocessing.ipynb`: Downloads the MIT-BIH dataset via `wfdb`, performs R-peak-centered segmentation (300 samples), applies AAMI EC57 binary mapping, and Min-Max normalization.
2. `02_Model_Training.ipynb`: Defines the MedNet-Lite architecture, implements class-weighted CrossEntropyLoss, and trains the model using the AdamW optimizer.
3. `03_Evaluation_and_Explainability.ipynb`: Evaluates the model on the held-out test set, generates ROC/PR curves, computes quantitative attention metrics (RMSE, QRS energy), runs the ablation study, and exports the model to ONNX format.

## ⚙️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/MohamedAlobede86/MedNet-Lite.git
cd MedNet-Lite

2. Set up the environment
python -m venv venv
# On Windows use: venv\Scripts\activate
# On macOS/Linux use: source venv/bin/activate
pip install -r requirements.txt
3. Launch Jupyter Notebook
jupyter notebook

 Citation
If you find this code useful in your research, please consider citing our paper:
@article{abdulali2026mednetlite,
  title={MedNet-Lite: An Interpretable Ultra-Lightweight 1D-CNN with CBAM for Real-Time ECG Anomaly Screening in Resource-Constrained Edge Computing Environments},
  author={AbdulAli, Mohammed Mustafa},
  journal={Biomedical Signal Processing and Control},
  year={2026},
  publisher={Elsevier},
  note={Manuscript ID: BSPC-D-25-13749}
}
Contact
For any questions or collaborations, please contact:
Mohammed Mustafa AbdulAli
Higher Institute of Science and Technology, Musaid, Libya
Email: mohamdabdulali@proton.me
