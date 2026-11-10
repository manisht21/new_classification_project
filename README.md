# 📰 News Article Classification using Machine Learning

## 📌 Project Overview
This project implements a complete Machine Learning pipeline to classify news articles into categories using the AG News dataset. The solution is built entirely using Python scripts (.py files) and can be executed from the terminal.

The pipeline includes:
- Data preprocessing
- Feature engineering using TF-IDF
- Model training using Logistic Regression
- Model evaluation
- Streamlit web deployment

---

## 📂 Dataset Source
Dataset Used: AG News Dataset  
Source: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset  

The dataset contains news articles categorized into:
- World
- Sports
- Business
- Sci/Tech

---

## 🧠 Model Used
- TF-IDF Vectorizer  
- Logistic Regression Classifier  
- Implemented using scikit-learn  
- Final Accuracy: ~91%

---

## 📁 Folder Structure
news_classification_project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── config.py
│
├── models/
│   └── news_classifier.pkl
│
├── results/
│   └── metrics.txt
│
├── requirements.txt
├── README.md
├── main.py
└── app.py

---

## ▶️ How to Run the Project

### 1️⃣ Create Virtual Environment
python -m venv venv

Activate (Windows):
venv\Scripts\activate

### 2️⃣ Install Requirements
pip install -r requirements.txt

### 3️⃣ Run Full ML Pipeline
python main.py

This will:
- Preprocess data
- Train model
- Save model in models/
- Evaluate model
- Save metrics in results/
- Print final accuracy

---

## 🌐 Run Streamlit Web App
streamlit run app.py

Then open:
http://localhost:8501

Enter any news article and get predicted category.

Test case link for article predictions(https://drive.google.com/file/d/1wgQPuqSwVAa5hF7QE21Vx7Fn-5d1fa9v/view?usp=sharing)

---

## 📊 Final Results
- Model: Logistic Regression
- Feature Extraction: TF-IDF
- Accuracy: ~91%
- Fully working ML pipeline
- Streamlit deployment ready

---

## 🎥 Video Explanation
(https://drive.google.com/drive/folders/1RkBqrY3ehA2B2RYg5XqSr7LwPpBA1lIy?usp=sharing)

---

## 🚀 Key Learnings
- End-to-end ML workflow
- Text preprocessing
- TF-IDF feature engineering
- Model training & evaluation
- Saving pipeline with joblib
- Streamlit deployment
- Clean project architecture using .py files only
✔ Deployment added  

---

## 👨‍💻 Developed By
Manish Tiwari
