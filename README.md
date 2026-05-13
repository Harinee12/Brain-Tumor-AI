# Reliable Brain Tumor AI Platform

Reliability-Aware Explainable Brain Tumor Detection System using Ensemble Deep Learning, Monte Carlo Dropout, GradCAM, and Streamlit Deployment.

---

## Live Application

https://reliable-brain-tumor-ai.streamlit.app/

---

## Overview

This project presents an AI-powered framework for automated brain tumor classification using MRI images. The system combines Ensemble Deep Learning, Explainable AI, and Reliability Analysis to improve prediction robustness and interpretability.

The framework integrates EfficientNet and ResNet18 models for classification and uses Monte Carlo Dropout for uncertainty estimation and GradCAM for visual explainability.

---

## Features

- Brain Tumor MRI Classification
- Ensemble Learning using EfficientNet and ResNet18
- Confidence Score Estimation
- Monte Carlo Dropout
- Uncertainty Analysis
- Reliability-Aware AI Framework
- GradCAM Explainability
- Interactive Streamlit Dashboard
- Multi-page Application
- PDF Report Generation
- Cloud Deployment using Streamlit Cloud

---

## Tumor Classes

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

---

## Technologies Used

- Python
- PyTorch
- Streamlit
- EfficientNet
- ResNet18
- OpenCV
- NumPy
- Matplotlib
- GradCAM

---

## System Workflow

```text
MRI Image
   ↓
Image Preprocessing
   ↓
EfficientNet + ResNet Ensemble
   ↓
Prediction Generation
   ↓
Confidence Estimation
   ↓
Monte Carlo Dropout
   ↓
Uncertainty Analysis
   ↓
GradCAM Explainability
   ↓
Reliability Evaluation
Project Structure
Brain-Tumor-AI/
│
├── app.py
├── utils.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── efficientnet_model.pth
│   └── resnet_model.pth
│
├── pages/
│   ├── 1_Confusion_Matrix.py
│   ├── 2_Model_Insights.py
│   ├── 3_Project_Overview.py
│   └── 4_Download_Report.py
│
└── .streamlit/
    └── config.toml
Run Locally

Clone the repository:

git clone https://github.com/Harinee12/Brain-Tumor-AI.git

Navigate to the project directory:

cd Brain-Tumor-AI

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py
Future Enhancements
GradCAM++
Improved Explainability
Larger MRI Dataset Training
Clinical Decision Support Features
Docker Deployment
FastAPI Integration

