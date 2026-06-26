# 🩺 Pneumonia Detection from Chest X-rays using Deep Learning

An Explainable AI system for automated pneumonia detection from chest X-ray images using **DenseNet121**, **Transfer Learning**, and **Grad-CAM** visualization. This project is being developed as an end-to-end medical imaging application with future integration of **Medical Visual Question Answering (VQA)**.

---

# 📌 Project Overview

Pneumonia is one of the leading causes of respiratory-related deaths worldwide. Early diagnosis from chest X-rays is essential, especially in hospitals with limited access to experienced radiologists.

This project develops an AI-assisted diagnostic system capable of:

* Detecting pneumonia from chest X-ray images.
* Highlighting suspicious lung regions using Grad-CAM.
* Providing confidence scores for predictions.
* Serving as the foundation for an interactive Medical AI assistant.

The long-term objective is to build a research-grade Explainable Medical AI system suitable for deployment and academic research.

---

# 🎯 Objectives

* Detect Pneumonia from Chest X-rays.
* Build an Explainable AI model using Grad-CAM.
* Develop an interactive Gradio web application.
* Extend the system to multiple thoracic diseases.
* Integrate Medical Visual Question Answering (LLaVA-Med / BLIP-2).

---

# 🚀 Key Features

✅ Binary Classification (Normal vs Pneumonia)

✅ Transfer Learning using DenseNet121

✅ Image Preprocessing & Data Augmentation

✅ Explainable AI using Grad-CAM

✅ Confidence Score Prediction

✅ Interactive Gradio Web Application (Upcoming)

✅ Medical Visual Question Answering (Upcoming)

---

# 🏥 Why This Project Matters

* Pneumonia causes millions of deaths every year.
* Early diagnosis significantly improves patient outcomes.
* Many healthcare centers lack experienced radiologists.
* Explainable AI increases clinician trust in AI predictions.
* Demonstrates practical application of AI in healthcare.

---

# 📂 Dataset

**Dataset:** Chest X-ray Pneumonia Dataset

**Source:** Kaggle

Total Images: **5,856**

Classes:

* NORMAL
* PNEUMONIA

Image Resolution:

224 × 224 pixels

Dataset Split:

* Training
* Validation
* Testing

---

# 🧠 Model Architecture

Input Chest X-ray (224×224×3)

↓

Data Augmentation

↓

DenseNet121 (ImageNet Pretrained)

↓

Global Average Pooling

↓

Dropout

↓

Sigmoid Output Layer

↓

Prediction

↓

Grad-CAM Heatmap

---

# 🛠️ Tech Stack

Programming Language

* Python

Deep Learning

* TensorFlow
* Keras

Computer Vision

* OpenCV

Machine Learning

* Scikit-learn

Visualization

* Matplotlib
* Seaborn

Deployment

* Gradio (Upcoming)

Version Control

* Git & GitHub

Development Environment

* Google Colab

---

# 📊 Model Performance

## Validation Accuracy

**92.9%**

## Test Accuracy

**87.66%**

## ROC-AUC Score

**0.945**

## Classification Report

| Class     | Precision | Recall | F1-score |
| --------- | --------: | -----: | -------: |
| Normal    |      0.89 |   0.77 |     0.82 |
| Pneumonia |      0.87 |   0.94 |     0.91 |

Overall Accuracy: **87.66%**

---

# 📈 Confusion Matrix

| Actual \ Predicted | Normal | Pneumonia |
| ------------------ | -----: | --------: |
| Normal             |    180 |        54 |
| Pneumonia          |     23 |       367 |

The model achieves a high recall for pneumonia, making it suitable as an initial screening system where minimizing missed disease cases is essential.

---

# 🔥 Explainable AI

Grad-CAM is used to visualize the lung regions responsible for the model's prediction.

This improves model interpretability by highlighting clinically relevant areas rather than providing only a classification label.

---

# 🌐 Future Enhancements

* Multi-class thoracic disease detection
* Medical Visual Question Answering (LLaVA-Med)
* Clinical report generation
* DICOM image support
* Cloud deployment
* REST API
* Docker containerization

---

# 📁 Project Structure

```
Medical_Xray_Project/

├── notebooks/
├── models/
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── gradcam/
├── app/
├── report/
├── images/
└── README.md
```

---

# 📚 Future Roadmap

* [x] Dataset Collection
* [x] Data Exploration
* [x] Data Visualization
* [x] Data Augmentation
* [x] DenseNet121 Training
* [x] Model Evaluation
* [ ] Sample Predictions
* [ ] Grad-CAM Explainability
* [ ] Gradio Web Application
* [ ] Medical Visual Question Answering
* [ ] Multi-Disease Classification
* [ ] Deployment

---

# 👩‍💻 Author

**Nahid Kausar**,
**Ananaya**

B.Tech Computer Science & Engineering

Passionate about Artificial Intelligence, Medical Imaging, Deep Learning, and Explainable AI.

---

# ⭐ Acknowledgements

* Kaggle Chest X-ray Pneumonia Dataset
* TensorFlow & Keras
* Google Colab
* OpenCV
* Scikit-learn
* Matplotlib
