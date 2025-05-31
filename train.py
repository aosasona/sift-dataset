# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-learn==1.5.1",
#     "tensorflow-macos==2.12.0",
#     "coremltools>=8.3.0",
# ]
# ///
import json

import coremltools as ct
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (LSTM, Dense, Embedding,
                                     GlobalAveragePooling1D, TextVectorization)
from tensorflow.keras.models import Sequential


def coreml_convert():
    with open("label_map.json") as f:
        labels = json.load(f)

    mlmodel = ct.convert(
        "articles_classifier",
        source="tensorflow",
        inputs=[ct.TensorType(shape=(1, 40), dtype=np.int32)],
        classifier_config=ct.ClassifierConfig(class_labels=labels),
        compute_units=ct.ComputeUnit.ALL,  # Enables GPU/Neural Engine
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15
    )
    mlmodel.save("ArticlesClassifier.mlpackage")
    print("✅ Saved CoreML model")


def tflite_convert():
    # Configure converter
    converter = tf.lite.TFLiteConverter.from_saved_model("articles_classifier")
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,     # TFLite native ops
        tf.lite.OpsSet.SELECT_TF_OPS        # Allow TF ops fallback
    ]

    # Optional: optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert
    tflite_model = converter.convert()

    # Save
    with open("articles_classifier.tflite", "wb") as f:
        f.write(tflite_model)

    print("✅ Saved TFLite model")


# Load the processed dataset
df = pd.read_csv('./dataset.csv')

x = df["text"].values
y = df["label"].values
x_train_raw, x_val_raw, y_train, y_val = train_test_split(
    x, y, test_size=0.2, stratify=df["label"].values
)

# Params
MAX_TOKENS = 12_000
OUTPUT_SEQUENCE_LEN = 40
BATCH_SIZE = 64
EPOCHS = 5
LABELS_LEN = df["label"].nunique()

# Make sure x_train_raw is a list/array of strings
x_train_raw = x_train_raw.astype(str)
x_val_raw = x_val_raw.astype(str)
vectorizer = TextVectorization(
    max_tokens=MAX_TOKENS,
    output_sequence_length=OUTPUT_SEQUENCE_LEN
)
vectorizer.adapt(x_train_raw)

# Tokenise the text
x_train = vectorizer(x_train_raw)
x_val = vectorizer(x_val_raw)

# Convert to numpy arrays for compatibility (with TFLite)
x_train = np.array(x_train)
x_val = np.array(x_val)

# Create the model
model = Sequential()
model.add(Embedding(
    input_dim=MAX_TOKENS,
    output_dim=128,
    input_length=OUTPUT_SEQUENCE_LEN
))
model.add(LSTM(64, return_sequences=True))
model.add(GlobalAveragePooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dense(LABELS_LEN, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# This is to ensure that the model stops training when the
# validation loss stops improving
early_stop = EarlyStopping(
    monitor="val_loss", patience=1, restore_best_weights=True
)

model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    verbose=1,
    callbacks=[early_stop]
)

# model.save("./articles_classifier.keras")
model.save("articles_classifier", save_format="tf")
print("Model trained and saved successfully!")

# Export vocab
vocab = vectorizer.get_vocabulary()
vocab_dict = {word: i for i, word in enumerate(vocab)}

# Export tokenizer metadata
tokenizer_info = {
    "vocab": vocab_dict,
    "max_length": OUTPUT_SEQUENCE_LEN,
    "pad_token": "[PAD]",
    "unk_token": "[UNK]"
}

# Save tokenizer to file
with open("tokenizer.json", "w") as f:
    json.dump(tokenizer_info, f, indent=2)

# Export the tflite model
tflite_convert()

# Export the model to coreml format
coreml_convert()
