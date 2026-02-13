# src/data_preprocessing.py

import pandas as pd


def preprocess_data(train_path, test_path, output_train_path, output_test_path):

    # Load data
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    # Rename columns (AG News format)
    train_df.columns = ["label", "title", "description"]
    test_df.columns = ["label", "title", "description"]

    # Combine title + description
    train_df["text"] = train_df["title"] + " " + train_df["description"]
    test_df["text"] = test_df["title"] + " " + test_df["description"]

    # Keep only needed columns
    train_df = train_df[["label", "text"]]
    test_df = test_df[["label", "text"]]

    # Save processed files
    train_df.to_csv(output_train_path, index=False)
    test_df.to_csv(output_test_path, index=False)

    print("Preprocessing done.")
