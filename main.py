from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model


def main():

    train_path = "data/raw/train.csv"
    test_path = "data/raw/test.csv"

    processed_train_path = "data/processed/train_processed.csv"
    processed_test_path = "data/processed/test_processed.csv"

    model_path = "models/news_classifier.pkl"
    results_path = "results/metrics.txt"

    print("Starting preprocessing...")

    preprocess_data(
        train_path,
        test_path,
        processed_train_path,
        processed_test_path
    )

    print("Preprocessing completed.")

    print("Starting training...")
    train_model(processed_train_path, model_path)
    print("Training completed.")

    print("Starting evaluation...")
    evaluate_model(processed_test_path, model_path, results_path)
    print("Evaluation completed.")

    print("Pipeline finished successfully!")


if __name__ == "__main__":
    main()
