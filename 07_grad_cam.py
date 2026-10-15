
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ==================================================
# Lesson 07: Explainable AI (Grad-CAM)
# ==================================================

# Explainable AI (XAI):
# Helps us understand what influences model predictions.
#
# Grad-CAM = Gradient-weighted Class Activation Mapping.
# Heatmap = highlights influential image regions.
# Feature Map = features extracted by CNN layers.
# Gradient = measures how a class score changes.
# Target Layer = CNN layer selected for explanation.


# ==================================================
# 1. Load Pretrained CNN
# ==================================================

# ResNet18 = pretrained CNN using skip connections.
# DEFAULT = pretrained ImageNet weights.
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze pretrained model weights.
# requires_grad=False = do not calculate weight gradients.
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace final FC layer for 7 skin lesion categories.
# The new FC layer has random weights.
model.fc = nn.Linear(model.fc.in_features, 7)

# eval() = set evaluation mode.
model.eval()

# Target layer = last residual block in ResNet18.
target_layers = [model.layer4[-1]]

# HAM10000 diagnosis categories.
classes = [
    "akiec", "bcc", "bkl", "df",
    "mel", "nv", "vasc"
]


# ==================================================
# 2. Load Skin Lesion Image
# ==================================================

# Path = helps locate files.
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

# Read first image ID from metadata.
with open(
    data_dir / "HAM10000_metadata.csv",
    "r",
    encoding="utf-8-sig"
) as file:
    first_image = next(csv.DictReader(file))

image_id = first_image["image_id"]

# Find the image in Part 1 or Part 2.
image_path = None

for folder in [
    "HAM10000_images_part_1",
    "HAM10000_images_part_2"
]:
    path = data_dir / folder / f"{image_id}.jpg"

    if path.exists():
        image_path = path
        break

if image_path is None:
    raise FileNotFoundError(
        f"Image {image_id}.jpg not found inside data/"
    )

# Open image with 3 RGB channels.
image = Image.open(image_path).convert("RGB")


# ==================================================
# 3. Preprocess Image
# ==================================================

# Use preprocessing recommended for pretrained ResNet18.
transform = weights.transforms()

# unsqueeze(0) = add a batch dimension.
# requires_grad_(True) = enable input gradients for Grad-CAM.
input_tensor = transform(image).unsqueeze(0)
input_tensor.requires_grad_(True)

# Resize image for visualization.
# CAM output matches the spatial size of the input tensor.
height, width = input_tensor.shape[-2:]

display_image = image.resize((width, height))

# Convert image to NumPy with pixel values from 0 to 1.
rgb_image = np.array(
    display_image,
    dtype=np.float32
) / 255.0


# ==================================================
# 4. Generate CNN Prediction
# ==================================================

# no_grad() = prediction only; no gradient calculation.
with torch.no_grad():

    # CNN generates 7 raw classification scores.
    outputs = model(input_tensor)

    # argmax() = find highest-scoring diagnosis class.
    predicted_index = outputs.argmax(dim=1).item()

print("=== Skin Lesion Prediction ===")
print("Image ID:", image_id)
print("Actual Label:", first_image["dx"])
print("Predicted Label:", classes[predicted_index])


# ==================================================
# 5. Generate Grad-CAM Heatmap
# ==================================================

# ClassifierOutputTarget = choose class to explain.
targets = [ClassifierOutputTarget(predicted_index)]

# enable_grad() = allow gradient calculation.
# Grad-CAM uses gradients and CNN feature maps.
with torch.enable_grad():

    with GradCAM(
        model=model,
        target_layers=target_layers
    ) as cam:

        # Generate heatmap for the predicted class.
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

# Overlay heatmap on original image.
visualization = show_cam_on_image(
    rgb_image,
    grayscale_cam,
    use_rgb=True
)


# ==================================================
# 6. Display Results
# ==================================================

plt.figure(figsize=(10, 4))

# Original skin lesion image.
plt.subplot(1, 2, 1)
plt.imshow(display_image)
plt.title("Original Skin Lesion")
plt.axis("off")

# Grad-CAM visualization.
plt.subplot(1, 2, 2)
plt.imshow(visualization)
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.tight_layout()
plt.show()


# ==================================================
# 7. Summary
# ==================================================

print("\n=== Grad-CAM Summary ===")
print("Model: ResNet18")
print("Target Layer: layer4[-1]")
print("Input Shape:", input_tensor.shape)
print("Heatmap Shape:", grayscale_cam.shape)

print("\nNote: Skin lesion classifier is NOT trained.")
print("The heatmap is for educational purposes only.")
