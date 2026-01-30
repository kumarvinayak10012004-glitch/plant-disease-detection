# 🌿 Automated Plant Disease Detection Using Deep Learning

## 📌 Project Overview
Plant diseases significantly impact agricultural productivity and global food security. Traditional manual inspection methods are time-consuming, subjective, and error-prone.  
This project presents an **AI-powered plant disease detection system** that uses **Deep Learning and Computer Vision** to identify plant diseases from leaf images and provide **real-time predictions** via a **web application**.

The system is designed to be **lightweight, accurate, and user-friendly**, making it suitable for farmers, agri-tech startups, educational institutions, and rural communities.

---

## 🎯 Problem Statement
Develop a deep learning-based web application that:
- Accepts leaf images from users
- Predicts the plant disease accurately
- Displays confidence scores and visual explanations (Grad-CAM)
- Provides actionable treatment recommendations
- Is deployable on cloud platforms for real-time access

---

## 🧠 Skills Gained
- Image preprocessing & augmentation
- CNN architecture design
- Transfer Learning (ResNet, EfficientNet, VGG16)
- Model evaluation & visualization
- Grad-CAM explainability
- Web app development (Streamlit / Flask)
- Model deployment & cloud hosting

---

## 🏢 Domain
**Computer Vision | Deep Learning | Agriculture Technology (AgriTech)**

---

## 💼 Business Use Cases
- 👨‍🌾 **Farmers** – Instant disease detection from leaf images  
- 🚜 **Agri-Tech Startups** – AI-driven crop advisory systems  
- 🎓 **Education** – Teaching AI applications in agriculture  
- 🌍 **Rural Communities** – Affordable disease diagnosis tools  

---

## 🗂 Dataset
**PlantVillage Dataset (Kaggle)**  
- ~54,000 labeled images  
- 38 plant disease categories  
- Image formats: JPG / PNG  

### Dataset Features
- Leaf images of healthy and diseased plants  
- Labels like:
  - Tomato – Early Blight  
  - Tomato – Late Blight  
  - Corn – Healthy  

---

## ⚙️ Project Approach

### 1️⃣ Data Preparation
- Resize images to `224 × 224`
- Normalize pixel values `(0–1)`
- Apply augmentation:
  - Rotation
  - Horizontal & vertical flip
  - Zoom
  - Brightness adjustment

---

### 2️⃣ Model Development
- Baseline CNN for initial benchmarking
- Transfer Learning using:
  - ResNet50
  - EfficientNet
  - VGG16
- Fine-tuning final layers for higher accuracy
- Save trained model for inference (`.h5 / .pt`)

---

### 3️⃣ Model Evaluation
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report
- Grad-CAM visualization for model explainability

---

### 4️⃣ Web Application Development
Built using **Streamlit / Flask** with features:
- 📤 Image upload
- 🔍 Real-time disease prediction
- 📊 Confidence score display
- 🔥 Grad-CAM heatmap visualization
- 💡 Actionable recommendations  
  *(Example: “Detected: Tomato Late Blight → Apply Fungicide XYZ”)*

streamlit run src\app_streamlit.py

---

## 📊 Results
- ✅ Disease classification accuracy **>90%**
- ⚡ Real-time predictions with low latency
- 🌐 Cloud-hosted and mobile-friendly
- 🧠 Explainable AI using Grad-CAM
- 🌾 Practical treatment recommendations integrated

---

## 📈 Evaluation Metrics

### Model Metrics
- Accuracy
- Precision
- Recall
- F1-Score

### System Metrics
- Prediction latency
- App responsiveness

### Usability Metrics
- Ease of use
- Interface clarity
- User interaction flow

---

## 🧪 Technologies Used
- **Programming:** Python  
- **Deep Learning:** TensorFlow, Keras, PyTorch  
- **Models:** CNN, ResNet, EfficientNet, VGG16  
- **Web Framework:** Streamlit / Flask  
- **Visualization:** Matplotlib, Seaborn, Grad-CAM  
- **Domain:** Computer Vision, Agriculture AI  

---

## 📁 Project Deliverables
- 📜 Source Code (Preprocessing, Training, Evaluation, App)
- 🤖 Trained Model (`.h5 / .pt`)
- 🌐 Web Application (Streamlit / Flask)
- 📘 Project Report
- 📊 Confusion Matrix & Grad-CAM Outputs

---

## 📌 Project Guidelines
- Use transfer learning for optimal performance
- Store trained models separately for deployment
- Keep application lightweight and responsive
- Maintain clean documentation and modular code

---

## 👤 Author
**Vinayak Kumar**  
_Data Science | Machine Learning | Computer Vision_

---

## ⭐ If you like this project, give it a star!
