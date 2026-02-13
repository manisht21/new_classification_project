# src/feature_engineering.py

from sklearn.feature_extraction.text import TfidfVectorizer


def create_vectorizer():
    return TfidfVectorizer(
        stop_words="english",
        max_df=0.7
    )
