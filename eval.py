# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tensorflow-macos==2.12.0",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "matplotlib",
#     "seaborn"
# ]
# ///
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# === Load Tokenizer ===
with open("tokenizer.json") as f:
    tok = json.load(f)
    vocab = {k.lower(): v for k, v in tok["vocab"].items()}
    max_len = tok["max_length"]
    pad_index = vocab.get("[PAD]", 0)
    unk_index = vocab.get("[UNK]", 1)


def tokenize(texts: list[str]) -> np.ndarray:
    sequences = []
    for text in texts:
        text = re.sub(r"[^\w\s]", "", text.lower())
        text = re.sub(r"\s+", " ", text)
        tokens = text.strip().split()
        ids = [vocab.get(token, unk_index) for token in tokens]
        ids = ids[:max_len] + [pad_index] * max(0, max_len - len(ids))
        sequences.append(ids)
    return np.array(sequences, dtype=np.int32)


# === Load label map ===
with open("label_map.json") as f:
    label_names = json.load(f)

# === Load dataset and split ===
df = pd.read_csv("news_dataset.csv")
x = df["text"].astype(str).tolist()
y = df["label"].values
_, x_val_raw, _, y_val = train_test_split(x, y, test_size=0.2, stratify=y)

x_val = tokenize(x_val_raw)

model = tf.keras.models.load_model("news_classifier")

# === Predict ===
y_probs = model.predict(x_val)
y_pred = np.argmax(y_probs, axis=1)

# === Evaluate ===
print("\n📊 Classification Report:")
report = classification_report(y_val, y_pred, target_names=label_names)
print(report)

with open("evaluation/report.txt", "w") as f:
    f.write(report)

# === Confusion matrix ===
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm, annot=False, cmap="Blues",
    xticklabels=label_names, yticklabels=label_names
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("evaluation/confusion_matrix.png")
print("✅ Evaluation saved to evaluation/report.txt and evaluation/confusion_matrix.png")
