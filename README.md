# Sift dataset

This was for an Advance AI and Applications module.

## Pre-requisites

- uv
- python (>=3.11)

All scripts in this directory are executed using `uv` as in:

```sh
uv run ./process.py && uv run ./train.py
```

This takes care of all the dependencies automatically.

# Models

- The CoreML model is located in `./NewsClassifier.mlpackage`
- The Keras model is located in `./news_classifier/`
- The `tflite` (not known as LiteRT) model (require Tensorflow bindings and ops enabled) is located in `./news_classifier.tflite`

# Dataset

The original dataset is located at `./News_Category_Dataset_v3.json`, and the processed version is at `./news_dataset.csv`.
