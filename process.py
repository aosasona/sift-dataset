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

    if label in ["ARTS", "ARTS & CULTURE", "CULTURE & ARTS"]:
        return "ARTS & CULTURE"
    elif label in ["STYLE", "STYLE & BEAUTY"]:
        return "STYLE"
    elif label in ["PARENTING", "PARENTS"]:
        return "PARENTING"
    elif label in ["FOOD & DRINK", "TASTE"]:
        return "FOOD & DRINK"
    elif label in ["WELLNESS", "HEALTHY LIVING"]:
        return "WELLNESS"
    elif label in ["BLACK VOICES", "QUEER VOICES", "LATINO VOICES", "WOMEN"]:
        return "IDENTITY"
    elif label in ["WORLDPOST", "THE WORLDPOST", "WORLD NEWS"]:
        return "WORLD NEWS"
    elif label in ["ENVIRONMENT", "GREEN"]:
        return "ENVIRONMENT"
    elif label in ["MEDIA", "COMEDY"]:
        return "ENTERTAINMENT"
    elif label == "GOOD NEWS":
        return "IMPACT"
    elif label == "FIFTY":
        return "WELLNESS"
    elif label == "U.S. NEWS":
        return "POLITICS"
    elif label == "IMPACT":
        return "SOCIETY"
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


# Load the dataset
df = pd.read_json("./News_Category_Dataset_v3.json", lines=True)

# Load the tech dataset
df_tech = pd.read_json("./lobsters_dataset.json")

# Merge the datasets
df = pd.concat([df, df_tech], ignore_index=True)

# Create the excerpt column
df["text"] = df["headline"] + " " + df["short_description"]
df["text"] = df["text"].apply(clean_text)

# print(df["text"].str.split().str.len().describe())

# Drop other unused columns
df = df[["text", "category"]]

# Merge similar categories
df["category"] = df["category"].apply(merge_label)

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
df.to_csv("./news_dataset.csv", index=False)
print("Dataset processed and saved successfully!")
