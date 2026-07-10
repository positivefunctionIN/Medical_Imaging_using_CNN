# Explainable Medical Imaging for Pneumonia Detection

A deep learning project that compares multiple CNN architectures for automated **Pneumonia Detection** from Chest X-ray images. The project focuses on model comparison, explainable AI using **Grad-CAM**, and deployment with **Gradio** and **Hugging Face Spaces**.

---

## Project Overview

This project develops and compares different deep learning models to detect **Pneumonia** from Chest X-ray images. The objective is to evaluate each model's performance and explain its predictions using Grad-CAM, making the system more interpretable and reliable.

---

## Objectives

* Detect Pneumonia from Chest X-rays.
* Compare multiple CNN architectures.
* Improve feature extraction using CBAM attention.
* Visualize model predictions with Grad-CAM.
* Deploy the best model using Gradio and Hugging Face.

---

## Dataset

**Dataset:** Chest X-ray Pneumonia Dataset

**Source:** Kaggle

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

**Classes**

* NORMAL
* PNEUMONIA

---

## Models

| Model                        | Status         |
| ---------------------------- | -------------- |
| Custom CNN                   | ✅ Completed    |
| Hybrid CNN + CBAM            | ✅ Completed |
| ResNet50 (Transfer Learning) | ✅ Completed     |

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* Classification Report

---

## Explainable AI

Grad-CAM will be implemented for all three models to visualize the regions responsible for each prediction, enabling better model interpretability and comparison.

---

## Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn
* Google Colab

**Deployment (Planned)**

* Gradio
* Hugging Face Spaces

---

## 📂 Repository Structure

```text
medical_imaging_project/
│
├── README.md
├── requirements.txt
├── model_1_custom_cnn.ipynb
├── model_2_hybrid_cnn_cbam.ipynb
├── model_3_resnet50.ipynb
├── comparison_report.ipynb
├── outputs/
│   ├── heatmaps/
│   └── results/
└── app/
```

---

## 🚀 Project Workflow

```text
Chest X-ray
      │
      ▼
Preprocessing
      │
      ▼
Custom CNN
      │
      ▼
Hybrid CNN + CBAM
      │
      ▼
ResNet50
      │
      ▼
Performance Comparison
      │
      ▼
Grad-CAM
      │
      ▼
Gradio App
      │
      ▼
Hugging Face Deployment
```

---

## 📌 Current Progress

* ✅ Dataset Preparation
* ✅ Data Augmentation & Preprocessing
* ✅ Custom CNN
* ✅ Hybrid CNN + CBAM
* ✅ ResNet50
* ⏳ Grad-CAM
* ⏳ Gradio Deployment
* ⏳ Hugging Face Deployment

---
