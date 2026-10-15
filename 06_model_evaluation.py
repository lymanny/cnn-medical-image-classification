
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)

# ==================================================
# Lesson 06: Model Evaluation
# (Accuracy, Precision, Recall, F1, ROC-AUC)
# ==================================================

# Model Evaluation:
# Measures how accurately a model predicts unseen data.
#
# Accuracy = overall percentage of correct predictions.
# Precision = how many predicted positives are correct.
# Recall = how many actual positives are detected.
# F1-score = balance between Precision and Recall.
# ROC = Receiver Operating Characteristic curve.
# AUC = Area Under the ROC Curve.
#
# This lesson uses simulated predictions, not a trained CNN.


# ==================================================
# 1. Create Example Predictions
# ==================================================

# Three HAM10000 diagnosis categories:
# 0 = Melanoma (mel)
# 1 = Melanocytic nevi (nv)
# 2 = Basal cell carcinoma (bcc)

classes = ["mel", "nv", "bcc"]

# Actual = correct diagnosis labels.
actual = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

# Scores = example probabilities for each class.
# Each row represents one image.
# Each row sums to 1.
scores = torch.tensor([
    [0.80, 0.15, 0.05],
    [0.10, 0.85, 0.05],
    [0.10, 0.20, 0.70],
    [0.40, 0.50, 0.10],
    [0.10, 0.70, 0.20],
    [0.10, 0.15, 0.75],
    [0.75, 0.20, 0.05],
    [0.20, 0.60, 0.20],
    [0.55, 0.15, 0.30],
    [0.65, 0.25, 0.10]
])

# argmax = find the class with the highest score.
predicted = scores.argmax(dim=1)


# ==================================================
# 2. Calculate Evaluation Metrics
# ==================================================

# numpy() = convert tensors to NumPy arrays.
y_true = actual.numpy()
y_pred = predicted.numpy()
y_scores = scores.numpy()

# Calculate overall accuracy.
accuracy = accuracy_score(y_true, y_pred)

# average="macro" = calculate each class equally.
precision = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)

recall = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)

f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)

# Multi-class ROC-AUC using One-vs-Rest.
# Requires scores for every class.
auc = roc_auc_score(
    y_true,
    y_scores,
    multi_class="ovr",
    average="macro",
    labels=[0, 1, 2]
)

print("=== Model Evaluation ===")
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-score:  {f1:.2%}")
print(f"ROC-AUC:   {auc:.4f}")


# ==================================================
# 3. Confusion Matrix
# ==================================================

# Confusion Matrix = shows classification mistakes.
cm = confusion_matrix(
    y_true, y_pred, labels=[0, 1, 2]
)

print("\n=== Confusion Matrix ===")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

display.plot(cmap="Blues", values_format="d")
plt.title("Skin Lesion Confusion Matrix")
plt.show()


# ==================================================
# 4. ROC Curve for Melanoma
# ==================================================

# One-vs-Rest:
# Treat melanoma as positive (1),
# and the other two categories as negative (0).

melanoma_true = (actual == 0).int().numpy()

# Probability assigned to melanoma.
melanoma_scores = y_scores[:, 0]

# Calculate false positive and true positive rates.
fpr, tpr, thresholds = roc_curve(
    melanoma_true, melanoma_scores
)

melanoma_auc = roc_auc_score(
    melanoma_true, melanoma_scores
)

# Plot melanoma ROC curve.
plt.plot(
    fpr, tpr,
    label=f"Melanoma AUC = {melanoma_auc:.2f}"
)

plt.plot([0, 1], [0, 1], "--", label="Random baseline")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Melanoma vs Rest")
plt.legend()
plt.grid(True)
plt.show()
