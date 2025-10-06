#  Prompted Segmentation for Drywall Quality Assurance

This repository contains the code and models for a project focused on **automated quality assurance of drywall**.  
The primary goal is to **segment two key features** from images:
- **Cracks**
- **Taping areas**

The project includes data preparation scripts, multiple model training experiments, and a **fully interactive web application** to demonstrate the final model.

---

##  Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Final Model Performance](#final-model-performance)
- [Directory Structure](#directory-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Model Development and Experiments](#model-development-and-experiments)

---

##  Project Overview

The core of this project is a **machine learning pipeline** that takes an image of drywall as input and produces a **binary mask** highlighting defects (cracks) or features (taping).  
This is achieved by **fine-tuning a state-of-the-art segmentation model** on a custom, augmented dataset.

The final deliverable is an **interactive web application** that allows users to:
- Upload their own images
- Provide point prompts
- Receive a segmentation mask generated in real time by the **fine-tuned Segment Anything Model 2.1 (SAM 2.1)**

---

## ⚙️ Features

- **🖥️ Interactive Web UI:** Built with React + TypeScript for easy image upload and interaction.  
- **⚡ High-Performance Backend:** Flask (Python) backend serving the fine-tuned SAM 2.1 model for fast inference.  
- **🎯 Point-Prompted Segmentation:** Leverages SAM 2.1 to generate precise masks from simple point grids.  
- **🔁 Comprehensive Data Pipeline:** Includes scripts for converting COCO annotations and performing extensive offline data augmentation.  
- ** Multi-Model Experimentation:** Explores five different architectures, with the best-performing one selected for deployment.  

---

##  Final Model Performance

The deployed model is a **fine-tuned SAM 2.1 (Hiera Tiny)** version.  
It was selected after a comparative analysis of five approaches.

| Metric | Test Set Score |
|--------|----------------|
| **Mean IoU (mIoU)** | `0.6407` |
| **Dice Score** | `0.7747` |

---

##  Directory Structure

```

/
├── backend/
│   ├── app.py                         # Flask server application
│   ├── model.py                       # Model loading and prediction logic
│   ├── requirements.txt               # Python dependencies
│   ├── sam2.1_hiera_tiny.pt           # Base SAM 2.1 model weights
│   └── sam2.1_best_e6_miou0.6201.pt   # Fine-tuned model weights
│
├── components/                        # React components for the UI
│   ├── ImageUploader.tsx
│   ├── PromptControls.tsx
│   └── ResultDisplay.tsx
│
├── services/                          # Frontend service for API calls
│   └── geminiService.ts
│
├── App.tsx                            # Main React application component
├── index.html                         # Entry point for the web app
├── package.json                       # Node.js project configuration
└── README.md                          # This file

````

---

## 🛠️ Setup and Installation

### Prerequisites
- Python **3.9+**
- Node.js and npm (or yarn)
- PyTorch and Torchvision (see `backend/requirements.txt`)

---

### 1️ Backend Setup

Set up the Python backend server:

```bash
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Ensure model weights are in this directory
# sam2.1_hiera_tiny.pt
# sam2.1_best_e6_miou0.6201.pt
````

---

### 2️⃣ Frontend Setup

From the project root:

```bash
# Install Node.js dependencies
npm install

# (Optional) Create a .env file for environment variables
```

---

##  Usage

### 1. Start the Backend Server

```bash
cd backend
python app.py
```

Backend runs at: **[http://localhost:5000](http://localhost:5000)**

---

### 2. Start the Frontend Application

```bash
npm run dev
```

Frontend runs at: **[http://localhost:3000](http://localhost:3000)**

---

### How to Use the App

1. **Upload an Image:** Click or drag a drywall image into the upload area.
2. **Provide a Prompt:** The app automatically generates a point grid—just click **"Generate Mask"**.
3. **View Results:** The original image and segmentation mask are displayed side-by-side.

---

##  Model Development and Experiments

This project involved a comprehensive investigation into optimal model architectures.
The dataset was formed by combining **two Roboflow datasets**:

* `drywall-join-detect`
* `cracks-3ii36`

After augmentation, the total dataset contained **~18.5k images**.

---

### 🧮 Model Comparison

| Approach | Model                   | Prompting       | Final mIoU | Final Dice |
| -------- | ----------------------- | --------------- | ---------- | ---------- |
| 1        | CLIPSeg                 | Text            | 0.5625     | 0.7106     |
| 2        | **SAM 2.1 (Final)**     | Point Grid      | **0.6407** | **0.7747** |
| 3        | SAM 2.1 (Improved Loss) | Point Grid      | 0.7024     | 0.8016     |
| 4        | SegFormer B2            | None (Semantic) | 0.6591     | 0.7706     |
| 5        | YOLOE-L                 | None (Semantic) | 0.5351     | 0.6683     |

> ⚠️ While **Approach 3** achieved the highest validation metrics, its model weights were lost.
> Therefore, the **robust and high-performing SAM 2.1 model (Approach 2)** was chosen for deployment.

---
