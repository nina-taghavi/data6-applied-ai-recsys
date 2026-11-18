import os
import pickle
import yaml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def main():
    # Read parameters from params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]

    seed = params["seed"]
    max_features = params["max_features"]
    solver = params["solver"]

    # Load prepared training data
    train_df = pd.read_csv("data/prepared/train.tsv", sep="\t")
    X_train = train_df["text"]
    y_train = train_df["label"]

    # Build model pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features)),
        ("classifier", LogisticRegression(solver=solver, random_state=seed))
    ])

    # Train model
    model.fit(X_train, y_train)

    # Save model
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved to models/model.pkl.")


if __name__ == "__main__":
    main()
