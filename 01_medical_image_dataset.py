
import csv
from pathlib import Path
from collections import Counter

# ==================================================
# Lesson 01: Medical Image Datasets (ISIC / HAM10000)
# ==================================================

# Medical Image Dataset:
# Collection of medical images and their labels.
#
# ISIC = International Skin Imaging Collaboration.
# HAM10000 = Dataset containing 10,015 skin lesion images.
# Metadata = Information describing each image.
#
# image_id = Unique image identifier.
# lesion_id = Identifier of the same skin lesion.
# dx = Diagnosis label.


# ==================================================
# 1. Skin Lesion Diagnosis Labels
# ==================================================

# HAM10000 contains 7 skin lesion categories.
#
# akiec = Actinic keratoses / intraepithelial carcinoma (precancerous or early cancerous changes).
# bcc = Basal cell carcinoma (skin cancer).
# bkl = Benign keratosis-like lesions (noncancerous).
# df = Dermatofibroma (usually harmless skin growth).
# mel = Melanoma (potentially serious skin cancer).
# nv = Melanocytic nevi (common skin moles).
# vasc = Vascular lesions (blood vessel lesions).

# Dictionary = stores diagnosis codes and their full names.
diagnosis_labels = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions"
}


# ==================================================
# 2. Load Dataset Metadata
# ==================================================

# Path = helps locate files on your computer.
base_dir = Path(__file__).resolve().parent

# CSV = file containing rows and columns of information.
csv_path = base_dir / "data" / "HAM10000_metadata.csv"

# Check whether the CSV file exists.
if not csv_path.exists():
    raise FileNotFoundError(
        "HAM10000_metadata.csv not found inside data/"
    )

# DictReader = reads CSV rows as dictionaries.
with open(csv_path, "r", encoding="utf-8-sig") as file:
    metadata = list(csv.DictReader(file))


# ==================================================
# 3. Explore Dataset
# ==================================================

# len() = counts the number of image records.
print("=== HAM10000 Dataset ===")
print("Total Images:", len(metadata))

# [0] = first image record.
first_image = metadata[0]

print("\n=== First Image ===")
print("Image ID:", first_image["image_id"])
print("Lesion ID:", first_image["lesion_id"])
print("Diagnosis Code:", first_image["dx"])

# get() = finds the full name of the diagnosis.
print(
    "Diagnosis Name:",
    diagnosis_labels.get(first_image["dx"], "Unknown")
)


# ==================================================
# 4. Count Skin Lesion Categories
# ==================================================

# Counter = counts the images in each category.
diagnoses = [row["dx"] for row in metadata]
counts = Counter(diagnoses)

print("\n=== Skin Lesion Categories ===")

# items() = gets each diagnosis code and its image count.
for code, count in sorted(counts.items()):
    name = diagnosis_labels.get(code, "Unknown")
    print(f"{code}: {name} = {count} images")


# ==================================================
# 5. Check Lesion Information
# ==================================================

# One lesion may have multiple images.
# Keep images of the same lesion in one dataset split
# to help prevent data leakage.

# Collect all lesion identifiers.
lesion_ids = [row["lesion_id"] for row in metadata]

# set() = removes duplicate lesion identifiers.
print("\n=== Lesion Information ===")
print("Total Images:", len(lesion_ids))
print("Unique Lesions:", len(set(lesion_ids)))
