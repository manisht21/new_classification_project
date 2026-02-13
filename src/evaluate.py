# src/evaluate.py

import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(processed_test_path, model_path, results_path):

    # Load processed test data
    df = pd.read_csv(processed_test_path)

    X_test = df["text"]
    y_test = df["label"]

    # Load saved model
    vectorizer, model = joblib.load(model_path)

    X_test_vectorized = vectorizer.transform(X_test)

    predictions = model.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    # Save results
    with open(results_path, "w") as f:
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write(report)

    print("Evaluation complete.")
    print("Accuracy:", accuracy)
