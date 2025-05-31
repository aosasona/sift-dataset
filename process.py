# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "scikit-learn",
# ]
# ///
import json
import re

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_text(text: str) -> str:
    # Lowercase
    text = text.lower()
    # Remove punctuation (keep only alphanumerics and whitespace)
    text = re.sub(r"[^\w\s]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Load the tech dataset
df = pd.read_json("./lobsters_dataset.json")

# Create the excerpt column
df["text"] = df["headline"] + " " + df["short_description"]
df["text"] = df["text"].apply(clean_text)

# print(df["text"].str.split().str.len().describe())

# Drop other unused columns
df = df[["text", "category"]]

# Merge similar categories
df["category"] = df["category"]

# Encode the category column
le = LabelEncoder()
df["label"] = le.fit_transform(df["category"])
LABELS_LEN = len(le.classes_)

# Save the label encoder for later use
le.classes_.tolist()
le_path = "./label_map.json"
with open(le_path, "w") as f:
    json.dump(le.classes_.tolist(), f)

# Save the processed dataset
df.to_csv("./dataset.csv", index=False)
print("Dataset processed and saved successfully!")
