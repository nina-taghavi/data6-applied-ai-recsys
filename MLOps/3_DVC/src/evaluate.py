import json
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def main():
    # Load model
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load test data
    test_df = pd.read_csv("data/prepared/test.tsv", sep="\t")
    X_test = test_df["text"]
    y_test = test_df["label"]

    # Predict
    y_pred = model.predict(X_test)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "accuracy": round(float(acc), 4),
        "f1": round(float(f1), 4),
        "test_samples": len(test_df)
    }

    # Save metrics
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluation completed. Metrics: {metrics}")


if __name__ == "__main__":
    main()
