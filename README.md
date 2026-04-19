<div align="center">

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>

# 🛡️ MobileNet4DFR

### **Enhanced Real-Time Deepfake Detection**

*Lightweight · Multi-Modal · Attention-Driven*

<br/>

> Inspired by the CVPRW 2024 paper  
> *"Faster Than Lies: Real-time Deepfake Detection using Binary Neural Networks"*  
> — reimagined for practical, real-world deployment.

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
[![Backbone](https://img.shields.io/badge/Backbone-MobileNetV3-brightgreen)](https://arxiv.org/abs/1801.04381)
[![Dataset](https://img.shields.io/badge/Datasets-CIFAKE%20%7C%20COCOFake-purple)](https://github.com/DanielNguyen-05/DeepfakeDetector)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Improvements & Novelties](#-key-improvements--novelties)
- [Architecture](#-architecture-overview)
- [Setup & Installation](#️-setup--installation)
- [Dataset Preparation](#-dataset-preparation)
- [Configuration & Training](#️-configuration--training)
- [Testing](#-testing-a-pretrained-model)
- [Project Structure](#-project-structure)

---

## 🔍 Overview

**MobileNet4DFR** is an enhanced deepfake detection framework that addresses the practical inference bottlenecks of Binary Neural Networks (BNNs) on standard hardware. By introducing **MobileNetV3-Small** as the core backbone and augmenting the feature pipeline with **Gabor Filters** and a **Squeeze-and-Excitation (SE) Attention Block**, this system achieves robust, real-time performance without specialized hardware accelerators.

---

## 🚀 Key Improvements & Novelties

### 1. 🧩 Lightweight Backbone — MobileNetV3-Small
Replaced the BNext architecture with **MobileNetV3-Small**, providing **true real-time inference** on standard GPUs and edge devices — no XNOR hardware accelerators required.

### 2. 🎨 Multi-Modal Feature Extraction

The model processes four complementary modalities from each input image:

| Modality | Description |
|---|---|
| 🖼️ **RGB** | Standard color image — baseline visual representation |
| 📡 **FFT** (Fast Fourier Transform) | Captures high-frequency artifacts and anomalous noise patterns from diffusion models |
| 🔲 **LBP** (Local Binary Patterns) | Extracts micro-texture anomalies in facial regions |
| 🌊 **Gabor Filters** | Multi-oriented edge & texture responses; sensitive to subtle facial interpolations (eyes, wrinkles) |

### 3. ⚡ Squeeze-and-Excitation (SE) Adapter
Before feeding the concatenated **6-channel input** (3 RGB + 1 FFT + 1 LBP + 1 Gabor) into the backbone, an SE Block **dynamically recalibrates channel-wise feature responses**. The model learns which modality — color, frequency, or texture — is most discriminative for each specific image.

---

## 🧠 Architecture Overview

```
Input Image (RGB)
 ├──> FFT Extraction   ───┐
 ├──> LBP Extraction   ───┼──> Concat (6 Channels) ──> [ SE-Block ] ──> Conv2D (3 Channels)
 └──> Gabor Filters    ───┘        (Attention)           (Compress)
                                                               │
                                                               ▼
                                                       [ MobileNetV3-Small ]
                                                               │
                                                               ▼
                                                    [ Linear Classifier ]
                                                               │
                                                    ┌──────────┴──────────┐
                                                    ✅ Real              ❌ Fake
```

---

## 🛠️ Setup & Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/DanielNguyen-05/DeepfakeDetector.git
cd DeepfakeDetector
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### Step 3 — Install Dependencies

Install PyTorch with the appropriate CUDA version (example for CUDA 12.x / modern RTX GPUs):

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> ⚠️ **Note:** Verify your CUDA version with `nvidia-smi` and choose the matching PyTorch wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## 📂 Dataset Preparation

This project supports evaluation on **CIFAKE** and **COCOFake**.

1. Create a `datasets/` folder in the project root.
2. Download and extract your chosen datasets.
3. Ensure the following directory structure:

```
MobileNet4DFR/
├── datasets/
│   ├── cifake/
│   │   ├── train/
│   │   └── test/
│   ├── coco2014/
│   └── fake_coco/
├── configs/
├── model.py
├── train.py
└── test.py
```

---

## ⚙️ Configuration & Training

### 🔧 Train the Model

```bash
python train.py --cfg configs/results_cifake_T_unfrozen.cfg
```

### 🌙 Train in Background (Recommended for long runs)

```bash
nohup python -u train.py --cfg configs/results_cifake_T_unfrozen.cfg > training_log.txt 2>&1 &
```

**Monitor GPU usage and training progress:**

```bash
nvidia-smi
tail -f training_log.txt
```

**Find the training process PID:**

```bash
ps -ef | grep train.py
```

**Force stop training (replace `1234567` with the actual PID):**

```bash
kill -9 1234567
```

---

## 🧪 Testing a Pretrained Model

Update the config file path as needed, then run:

```bash
python test.py --cfg configs/results_cifake_T_unfrozen.cfg
```

---

## 📁 Project Structure

```
MobileNet4DFR/
├── configs/                    # Training configuration files (.cfg)
├── datasets/                   # Dataset directory (not tracked by git)
│   ├── cifake/
│   ├── coco2014/
│   └── fake_coco/
├── model.py                    # MobileNet4DFR model definition
├── train.py                    # Training script
├── test.py                     # Evaluation / inference script
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Original paper: *"Faster Than Lies: Real-time Deepfake Detection using Binary Neural Networks"* — CVPRW 2024
- [MobileNetV3-Small](https://arxiv.org/pdf/1905.02244) — Howard et al., Google AI
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) — Hu et al.
- [CIFAKE Dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- [COCOFake Dataset](https://cocodataset.org/#download)

---

<div align="center">

Made with ❤️ by [DanielNguyen-05](https://github.com/DanielNguyen-05)

⭐ *If this project helps your research, please consider giving it a star!* ⭐

</div>