# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tensorflow-macos==2.12.0",
#     "numpy",
# ]
# ///
import json

import numpy as np
import tensorflow as tf

# Load saved model
model = tf.keras.models.load_model("articles_classifier")

# Load label map
with open("label_map.json") as f:
    label_names = json.load(f)

# Load tokenizer info
with open("tokenizer.json") as f:
    tokenizer_info = json.load(f)
    vocab = tokenizer_info["vocab"]
    max_length = tokenizer_info["max_length"]
    pad_index = vocab.get("[PAD]", 0)
    unk_index = vocab.get("[UNK]", 1)

# Lowercase vocab keys for consistency
vocab = {k.lower(): v for k, v in vocab.items()}


def tokenize(text: str) -> np.ndarray:
    """
    Tokenize and pad/truncate the input text using the exported vocab.
    """
    tokens = text.lower().split()  # Simple whitespace tokenization + lowercasing
    token_ids = [vocab.get(token, unk_index) for token in tokens]

    # Pad or truncate to max_length
    if len(token_ids) < max_length:
        token_ids += [pad_index] * (max_length - len(token_ids))
    else:
        token_ids = token_ids[:max_length]

    return np.array([token_ids], dtype=np.int32)  # Shape (1, max_length)


def predict_category(text: str) -> str:
    """
    Predict the category of a news article given its text.
    """
    input_tensor = tokenize(text)
    predictions = model.predict(input_tensor)
    predicted_index = np.argmax(predictions[0])
    predicted_label = label_names[predicted_index]
    return predicted_label


if __name__ == "__main__":
    # You can replace this with your own text or use input()
    # sample_text = "The government announced new tax reforms aimed at supporting small businesses and startups, promising reduced bureaucracy and increased funding access."

    sample_texts = [
        "Woman Who Called Cops On Black Bird-Watcher Loses Lawsuit Against Ex-Employer Amy Cooper accused investment firm Franklin Templeton of unfairly firing her and branding her a racist after video of the Central Park encounter went viral.",
        "Apple unveiled its latest M-series chip promising improved performance and battery life for the next generation of MacBook devices.",
        "Nutritionists emphasize the importance of fiber-rich diets to support gut health and reduce the risk of chronic diseases.",
        "The spring fashion week brought bold color palettes, oversized blazers, and a strong return to 90s-inspired streetwear.",
        "Iceland’s dramatic landscapes and geothermal spas continue to attract tourists seeking both adventure and relaxation.",
        "Experts recommend ten-minute mindfulness exercises as a way to combat daily stress and improve mental clarity.",
        "Lawmakers debated the new climate policy in parliament, with concerns raised over long-term economic impacts.",
        "The national team advanced to the semifinals after a dramatic penalty shootout that kept fans on the edge of their seats.",
        "The highly anticipated sequel to last year’s blockbuster has finally hit theaters, breaking opening weekend records worldwide.",
        "Chefs across the country are embracing fermented ingredients, citing both health benefits and flavor complexity.",
        "Researchers at MIT have developed a new AI model that can accurately predict protein folding, revolutionizing drug discovery.",
    ]

    for text in sample_texts:
        predicted_label = predict_category(text)
        print(f"Text: {text}\nPredicted Category: {predicted_label}\n")
