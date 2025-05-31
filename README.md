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

- The CoreML model is located in `./ArticlesClassifier.mlpackage`
- The Keras model is located in `./articles_classifier/`
- The `tflite` (not known as LiteRT) model (require Tensorflow bindings and ops enabled) is located in `./articles_classifier.tflite`

# Dataset

The original dataset is located at `./dataset.json`, and the processed version is at `./dataset.csv`.
