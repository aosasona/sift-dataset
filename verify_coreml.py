# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coremltools>=8.3.0",
# ]
# ///
import coremltools as ct

model = ct.models.MLModel("NewsClassifier.mlpackage")
print("Input(s):", model.input_description)
print("Output(s):", model.output_description)
