
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# ==================================================
# Lesson 02: Skin Lesion Image Preprocessing
# ==================================================

# Image Preprocessing:
# Prepares images before sending them into a CNN.
#
# Resize = changes image width and height.
# ToTensor = converts an image to a PyTorch tensor.
# Normalize = adjusts pixel values using mean and std.
# RGB = image with 3 color channels.
# Shape = [Channels, Height, Width].


# ==================================================
# 1. Load Skin Lesion Image
# ==================================================

# Path = locates files on the computer.
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

# Read the metadata CSV to get an image ID.
with open(data_dir / "HAM10000_metadata.csv",
          "r", encoding="utf-8-sig") as file:
    first_image = next(csv.DictReader(file))

image_id = first_image["image_id"]

# Images may be inside part_1 or part_2.
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
        f"Image {image_id}.jpg not found in data/"
    )

# Open the image and convert it to RGB.
image = Image.open(image_path).convert("RGB")

print("=== Original Image ===")
print("Image ID:", image_id)
print("Image Size:", image.size)
print("Image Mode:", image.mode)


# ==================================================
# 2. Preprocess Image
# ==================================================

# Compose = combines image preprocessing steps.
# Resize = changes image dimensions to 224x224.
# ToTensor = converts pixels to values from 0 to 1.
# Normalize = adjusts each RGB channel.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Apply preprocessing to the image.
image_tensor = transform(image)

print("\n=== Preprocessed Image ===")
print("Tensor Shape:", image_tensor.shape)
print("Tensor Type:", image_tensor.dtype)


# ==================================================
# 3. Visualize Original and Resized Images
# ==================================================

# Resize only for display.
resized_image = image.resize((224, 224))

plt.figure(figsize=(9, 4))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(resized_image)
plt.title("Resized Image (224x224)")
plt.axis("off")

plt.tight_layout()
plt.show()
