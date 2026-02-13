# 📰 News Article Classification using Machine Learning

## 📌 Project Overview
This project implements a complete machine learning pipeline to classify news articles into predefined categories using Python scripts only. The pipeline includes data preprocessing, feature engineering, model training, and evaluation, and can be executed directly from the terminal.

---

## 📂 Dataset Source

Dataset Used: AG News Dataset  
Source: Public Kaggle Dataset  
https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset  

The dataset contains news articles categorized into:
- World
- Sports
- Business
- Sci/Tech

---

## 📁 Folder Structure

news_classification_project/
│
├── data/
│   ├── raw/              # Original dataset files (train.csv, test.csv)
│   └── processed/        # Cleaned and processed dataset
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── config.py
│
├── models/
│   └── news_classifier.pkl   # Saved trained model
│
├── results/
│   └── metrics.txt           # Evaluation results
│
├── requirements.txt
├── README.md
└── main.py

---

## ⚙️ Steps to Run the Project

1️⃣ Clone the Repository

git clone https://github.com/manisht21/new_classification_project.git
cd new_classification_project

2️⃣ Create Virtual Environment (Optional but Recommended)

python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Run the Full Pipeline

python main.py

Running main.py will:
- Preprocess data
- Train the model
- Evaluate the model
- Save results
- Print final accuracy

---

## 🧠 Machine Learning Model Used

Feature Extraction: TF-IDF Vectorization  
Model: Logistic Regression  
Library: Scikit-learn  

Logistic Regression was chosen because:
- It performs well for text classification
- It is efficient and fast
- Works effectively with TF-IDF features

---

## 📊 Final Results

Accuracy: ~91.6%  
Confusion Matrix saved in results/metrics.txt  
Model saved in models/news_classifier.pkl  

---

## 🏗️ Machine Learning Workflow

1. Data Preprocessing
   - Lowercasing
   - Removing special characters
   - Handling missing values

2. Feature Engineering
   - Convert text into numerical vectors using TF-IDF

3. Model Training
   - Train Logistic Regression model
   - Save model using joblib

4. Model Evaluation
   - Accuracy calculation
   - Confusion matrix generation
   - Save results to file

---

## 🎥 Explanation Video

Video Explanation Link:
(Add your Google Drive or YouTube link here)

---

## 📌 Key Learnings

- Building end-to-end ML pipeline using only Python scripts.
- Text preprocessing and feature extraction.
- Model saving and loading.
- Proper project architecture for production-ready ML systems.
- Running ML workflow from terminal without Jupyter Notebook.
