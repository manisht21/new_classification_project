# src/train.py

import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from src.feature_engineering import create_vectorizer


def train_model(processed_train_path, model_path):

    # Load processed data
    df = pd.read_csv(processed_train_path)

    X = df["text"]
    y = df["label"]

    # Create vectorizer
    vectorizer = create_vectorizer()

    X_vectorized = vectorizer.fit_transform(X)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_vectorized, y)

    # Save model + vectorizer together
    joblib.dump((vectorizer, model), model_path)

    print("Model saved at:", model_path)
