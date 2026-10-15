
import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ==================================================
# Lesson 04: Build a Skin Lesion CNN Classifier
# ==================================================

# CNN = Convolutional Neural Network (complete model).
# CNN Classifier = CNN used to classify images.
# CNN Laye(Conv Layer) = extracts image features using filters.
# ReLU = helps learn complex patterns.
# MaxPool = reduces feature map size.
# Flatten = converts feature maps into a vector.
# FC Layer = Fully Connected layer for classification.
# Logits = raw prediction scores.
# Softmax = converts logits into probabilities.


# ==================================================
# 1. Define Skin Lesion Categories
# ==================================================

# HAM10000 contains 7 diagnosis categories.
classes = [
    "akiec", "bcc", "bkl", "df",
    "mel", "nv", "vasc"
]


# ==================================================
# 2. Build CNN Classifier
# ==================================================

# nn.Module = base class for PyTorch neural networks.
class SkinLesionCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Conv2d = Convolutional Layer.
        # 3 = RGB input channels.
        # 16 = output feature maps.
        # kernel_size=3 = 3x3 filter.
        # padding=1 = preserves image dimensions.
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, padding=1
        )

        # Second Conv Layer: 16 → 32 feature maps.
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, padding=1
        )

        # ReLU = activation function.
        self.relu = nn.ReLU()

        # MaxPool2d = reduces height and width by half.
        self.pool = nn.MaxPool2d(2, 2)

        # FC Layer = Fully Connected Layer.
        # 32 * 56 * 56 = flattened features.
        # 7 = output classes.
        self.fc = nn.Linear(32 * 56 * 56, 7)

    # forward() = defines how images pass through CNN.
    def forward(self, x):

        # Conv → ReLU → Pooling.
        x = self.pool(self.relu(self.conv1(x)))

        # Second Conv → ReLU → Pooling.
        x = self.pool(self.relu(self.conv2(x)))

        # Flatten feature maps into a vector.
        x = torch.flatten(x, start_dim=1)

        # FC Layer generates 7 classification scores.
        x = self.fc(x)

        return x


# ==================================================
# 3. Load Skin Lesion Image
# ==================================================

# Path = helps locate files.
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

# Read the first image ID from metadata.
with open(
    data_dir / "HAM10000_metadata.csv",
    "r",
    encoding="utf-8-sig"
) as file:
    first_image = next(csv.DictReader(file))

image_id = first_image["image_id"]

# Search for the image in both folders.
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

# Open image with 3 RGB channels.
image = Image.open(image_path).convert("RGB")


# ==================================================
# 4. Image Preprocessing
# ==================================================

# Compose = combines preprocessing steps.
# Resize = changes image size to 224x224.
# ToTensor = converts image into PyTorch tensor.
# Normalize = adjusts RGB pixel values.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# unsqueeze(0) = adds a batch dimension.
# [3, 224, 224] → [1, 3, 224, 224].
image_tensor = transform(image).unsqueeze(0)


# ==================================================
# 5. Forward Propagation
# ==================================================

# Create the CNN classifier.
model = SkinLesionCNN()

# eval() = switches model to evaluation mode.
model.eval()

# inference_mode() = disables gradient tracking.
with torch.inference_mode():

    # CNN generates 7 raw prediction scores.
    outputs = model(image_tensor)

    # Softmax = converts logits into probabilities.
    probabilities = torch.softmax(outputs, dim=1)

    # argmax() = finds class with highest probability.
    predicted_index = probabilities.argmax(dim=1).item()


# ==================================================
# 6. Display Results
# ==================================================

print("=== Skin Lesion CNN Classifier ===")
print("Image ID:", image_id)
print("Input Shape:", image_tensor.shape)
print("Output Shape:", outputs.shape)

print("\n=== Prediction ===")
print("Actual Label:", first_image["dx"])
print("Predicted Label:", classes[predicted_index])

print("\n=== Class Probabilities ===")

for name, probability in zip(classes, probabilities[0]):
    print(f"{name}: {probability.item():.2%}")

print("\nNote: Model is not trained yet.")
print("Predictions are not medically meaningful.")
