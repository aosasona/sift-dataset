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


def merge_label(label: str) -> str:
    label = label.strip().upper()

    if label in ["AI", "COGNITIVE SCIENCE"]:
        return "ARTIFICIAL INTELLIGENCE"

    elif label in ["PROGRAMMING", "DEVOPS", "OPERATING SYSTEMS", "DESIGN", "COMPUTER SCIENCE"]:
        return "ENGINEERING"

    elif label in ["CRYPTOGRAPHY", "CYBERSECURITY"]:
        return "SECURITY"

    elif label in ["SCIENCE", "MATHEMATICS"]:
        return "SCIENCE"

    elif label in ["FINANCE", "EDUCATION", "LAW"]:
        return "INDUSTRY"

    elif label == "HARDWARE":
        return "HARDWARE"

    elif label in ["GAMES", "ART"]:
        return "MEDIA"

    elif label in ["UPDATES", "RETROSPECTIVE"]:
        return "META"

    elif label in ["RANT", "SATIRE", "CULTURE"]:
        return "PERSPECTIVE"

    else:
        return label


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
# df["text"] = df["text"].apply(clean_text)

# print(df["text"].str.split().str.len().describe())

# Drop other unused columns
df = df[["text", "category"]]

# Merge similar categories
# df["category"] = df["category"]
df["category"] = df["category"].apply(merge_label)

# Encode the category column
le = LabelEncoder()
df["label"] = df["category"].astype(str).apply(lambda x: x.strip().upper())
df["label_int"] = le.fit_transform(df["category"])
LABELS_LEN = len(le.classes_)

# Save the label encoder for later use
le.classes_.tolist()
le_path = "./label_map.json"
with open(le_path, "w") as f:
    json.dump(le.classes_.tolist(), f)

# Save the processed dataset
df = df[["text", "label"]]
df.to_csv("./dataset.csv", index=False)
print("Dataset processed and saved successfully!")
