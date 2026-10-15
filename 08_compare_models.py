
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==================================================
# Lesson 08: Compare Models and Analyze Results
# ==================================================

# Model Comparison:
# Compares models using the same evaluation dataset.
#
# Custom CNN = convolutional network built from scratch.
# ResNet18 = CNN with residual (skip) connections.
# EfficientNet-B0 = efficient CNN architecture.
#
# Accuracy = percentage of all correct predictions.
# Precision = how often predicted classes are correct.
# Recall = how many actual cases are detected.
# F1-score = balances precision and recall.
#
# NOTE: These predictions are simulated, not from
# trained skin lesion classification models.


# ==================================================
# 1. Create Example Predictions
# ==================================================

# Three example diagnosis categories:
# 0 = Melanoma, 1 = Nevus, 2 = Basal cell carcinoma.

# Actual = correct labels for 10 images.
actual = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

# Each model predicts labels for the same images.
predictions = {
    "Custom CNN": np.array([0, 1, 2, 1, 1, 0, 0, 2, 2, 1]),
    "ResNet18": np.array([0, 1, 2, 0, 1, 2, 0, 2, 2, 0]),
    "EfficientNet-B0": np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 1])
}


# ==================================================
# 2. Calculate Evaluation Metrics
# ==================================================

# Dictionary = stores results for each model.
results = {}

# items() = gets model name and predictions.
for name, predicted in predictions.items():

    # Compare predictions with correct labels.
    accuracy = accuracy_score(actual, predicted)

    # macro = gives equal importance to each class.
    precision = precision_score(
        actual, predicted, average="macro", zero_division=0
    )

    recall = recall_score(
        actual, predicted, average="macro", zero_division=0
    )

    f1 = f1_score(
        actual, predicted, average="macro", zero_division=0
    )

    # Store calculated metrics.
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


# ==================================================
# 3. Display Comparison Results
# ==================================================

print("=== Model Comparison ===")

for name, metrics in results.items():
    print(f"\n{name}")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.2%}")


# ==================================================
# 4. Visualize Model Performance
# ==================================================

# Get model names and F1-scores.
names = list(results.keys())
f1_scores = [results[name]["F1"] * 100 for name in names]

# Plot F1-scores as a bar chart.
plt.bar(names, f1_scores)

plt.title("CNN Model Comparison (Simulated)")
plt.xlabel("Model")
plt.ylabel("Macro F1-score (%)")
plt.ylim(0, 100)

# Show scores above each bar.
for i, score in enumerate(f1_scores):
    plt.text(i, score + 1, f"{score:.1f}%", ha="center")

plt.tight_layout()
plt.show()


# ==================================================
# 5. Find Highest F1-score
# ==================================================

# max() = finds model with the highest F1-score.
best_model = max(
    results,
    key=lambda name: results[name]["F1"]
)

print("\n=== Highest Example F1-score ===")
print("Model:", best_model)
print(f"F1-score: {results[best_model]['F1']:.2%}")

print("\nNote: These are simulated results.")
