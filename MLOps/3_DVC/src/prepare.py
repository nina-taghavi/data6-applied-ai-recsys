import os
import sys
import xml.etree.ElementTree as ET
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # Read parameters from params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["prepare"]

    split = params["split"]
    seed = params["seed"]

    # Parse XML dataset
    input_path = "data/data.xml"
    tree = ET.parse(input_path)
    root = tree.getroot()

    records = []
    for row in root.findall("row"):
        text = row.attrib.get("Body", "")
        target = int(row.attrib.get("Target", "0"))
        records.append({"text": text, "label": target})

    df = pd.DataFrame(records)

    # Train-test split
    train_df, test_df = train_test_split(df, test_size=split, random_state=seed)

    # Output directory
    os.makedirs("data/prepared", exist_ok=True)
    train_df.to_csv("data/prepared/train.tsv", sep="\t", index=False)
    test_df.to_csv("data/prepared/test.tsv", sep="\t", index=False)

    print(f"Data prepared successfully: {len(train_df)} train, {len(test_df)} test.")


if __name__ == "__main__":
    main()
