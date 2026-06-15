 Pneumonia Detection from Chest X-rays using Deep Learning

> ResNet50-based system to detect pneumonia from chest X-ray images with Grad-CAM explainability

 Project Overview

This project builds a deep learning model to detect pneumonia from chest X-ray images. It uses transfer learning with ResNet50 and provides **heatmap visualizations** showing which lung regions indicate infection.

Why This Matters:
- Pneumonia causes ~2.5 million deaths annually worldwide
- Early detection saves lives, especially in rural areas with limited radiologists
- Explainable AI (Grad-CAM) helps doctors trust the diagnosis


Key Features
Binary Classification : Normal vs Pneumonia
Transfer Learning : ResNet50 pretrained on ImageNet 
Explainability : Grad-CAM heatmaps show infected lung regions 
Web Ready : Can be deployed with Streamlit 
High Sensitivity : Optimized to not miss sick patients 


Dataset
Source: [Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kaggle)

 Statistic  Value 
 Total images : 5,856 
 Image size : 224×224 pixels 
 Classes : 2 (Normal, Pneumonia) 
 Train/Validation/Test : 80/10/10 split 


 Sample Images:
 Normal Chest X-ray | Pneumonia Chest X-ray 
 Clear, dark lung fields | Hazy white patches (infiltrates) 
 Sharp heart borders | Blurred borders 
 No opacities | Ground-glass opacities 


 Model Architecture
ResNet50 (pretrained, frozen)
↓
GlobalAveragePooling2D
↓
Dense(128, activation='relu')
↓
Dropout(0.3)
↓
Dense(1, activation='sigmoid')

**Total Parameters:** ~23.5M (mostly from ResNet50 base)


 Results:
 Metric | Value 
 Test Accuracy | 85-90% 
 **Sensitivity (Recall)** | **>90%** (most important for medical) 
 Precision | ~85% 
 AUC-ROC | ~0.92 

 Confusion Matrix:
Predicted
Normal Pneumonia
Actual
Normal TN FP
Pneumonia FN TP

