import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_model(train_path, model_path):

    df = pd.read_csv(train_path)

    X = df["text"]
    y = df["label"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X, y)

    joblib.dump(pipeline, model_path)

    print("Model saved at:", model_path)
