
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# ==================================================
# Lesson 03: Data Augmentation
# ==================================================

# Data Augmentation:
# Creates modified versions of training images.
# Helps improve generalization and reduce overfitting.
#
# HorizontalFlip = flip left and right.
# VerticalFlip = flip top and bottom.
# Rotation = rotate an image.
# ColorJitter = change brightness, contrast, and color.


# ==================================================
# 1. Load Skin Lesion Image
# ==================================================

# Path = helps locate files on your computer.
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

# Read the first image ID from metadata.
with open(data_dir / "HAM10000_metadata.csv",
          "r", encoding="utf-8-sig") as file:
    first_image = next(csv.DictReader(file))

image_id = first_image["image_id"]

# Find image in either image folder.
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
        f"{image_id}.jpg not found inside data/"
    )

# Open the skin lesion image as RGB.
image = Image.open(image_path).convert("RGB")


# ==================================================
# 2. Define Data Augmentation
# ==================================================

# Compose = combines several image transformations.
# p=1.0 = always flip (for easy visualization).
# degrees=30 = randomly rotate between -30 and +30.
augmentation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1
    )
])


# ==================================================
# 3. Generate Augmented Images
# ==================================================

# Resize original image for comparison.
original = transforms.Resize((224, 224))(image)

# Each call generates another random transformation.
augmented_images = [
    augmentation(image) for _ in range(4)
]


# ==================================================
# 4. Display Images
# ==================================================

# Show original and four augmented versions.
images = [original] + augmented_images

titles = [
    "Original",
    "Augmented 1",
    "Augmented 2",
    "Augmented 3",
    "Augmented 4"
]

plt.figure(figsize=(12, 3))

for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
