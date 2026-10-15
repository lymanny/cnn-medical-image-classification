
import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

# ==================================================
# Lesson 05: Transfer Learning (ResNet / EfficientNet)
# ==================================================

# Transfer Learning:
# Reuses knowledge from an already trained CNN.
#
# Pretrained Model = CNN with previously learned weights.
# ResNet = CNN with skip connections.
# EfficientNet = CNN designed for efficiency.
# Freeze = stop weights from being updated.
# Fine-Tuning = train pretrained weights on a new task.
# FC Layer = Fully Connected classification layer.


# ==================================================
# 1. Define Skin Lesion Categories
# ==================================================

# HAM10000 contains 7 diagnosis categories.
classes = [
    "akiec", "bcc", "bkl", "df",
    "mel", "nv", "vasc"
]


# ==================================================
# 2. Load Pretrained ResNet18
# ==================================================

# DEFAULT = pretrained ImageNet weights.
# First run may download model weights.
weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

# Freeze pretrained layers.
# requires_grad=False = do not update these weights.
for parameter in model.parameters():
    parameter.requires_grad = False

# ResNet18 originally predicts 1000 ImageNet classes.
# Replace its final FC layer to predict 7 skin lesion classes.
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 7)

# The new FC layer remains trainable.
print("=== Pretrained ResNet18 ===")
print("Original FC Inputs:", in_features)
print("New Output Classes:", model.fc.out_features)


# ==================================================
# 3. Load HAM10000 Image
# ==================================================

# Path = locates project files.
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

# Read first image from metadata.
with open(
    data_dir / "HAM10000_metadata.csv",
    "r",
    encoding="utf-8-sig"
) as file:
    first_image = next(csv.DictReader(file))

image_id = first_image["image_id"]

# Find image in either dataset folder.
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
        f"Image {image_id}.jpg not found."
    )

# Open skin lesion image using RGB.
image = Image.open(image_path).convert("RGB")


# ==================================================
# 4. Preprocess Image
# ==================================================

# weights.transforms() = use preprocessing
# recommended for these pretrained weights.
transform = weights.transforms()

# unsqueeze(0) = add batch dimension.
# [3, 224, 224] → [1, 3, 224, 224].
image_tensor = transform(image).unsqueeze(0)


# ==================================================
# 5. Forward Propagation
# ==================================================

# eval() = set evaluation mode.
model.eval()

# inference_mode() = disable gradient tracking.
with torch.inference_mode():

    # ResNet18 generates 7 classification scores.
    outputs = model(image_tensor)

    # Softmax = convert raw scores to probabilities.
    probabilities = torch.softmax(outputs, dim=1)

    # argmax = index of highest score.
    predicted_index = probabilities.argmax(dim=1).item()


# ==================================================
# 6. Display Results
# ==================================================

print("\n=== Skin Lesion Image ===")
print("Image ID:", image_id)
print("Actual Label:", first_image["dx"])
print("Input Shape:", image_tensor.shape)

print("\n=== ResNet18 Output ===")
print("Output Shape:", outputs.shape)
print("Predicted Label:", classes[predicted_index])

print("\n=== Class Probabilities ===")

for name, probability in zip(classes, probabilities[0]):
    print(f"{name}: {probability.item():.2%}")

print("\nNote: The new FC layer has NOT been trained.")
print("Predictions are not medically meaningful.")
