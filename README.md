# 🏥 CNN Medical Image Classification

A beginner-friendly **PyTorch CNN project** exploring skin lesion image classification using the **HAM10000 dataset**, image preprocessing, data augmentation, custom CNNs, transfer learning, evaluation metrics, and explainable AI (Grad-CAM).

## 📚 Topics

### 1. 🗂️ Medical Image Datasets (ISIC / HAM10000)

Learn how to explore medical image datasets, understand skin lesion diagnosis labels, and analyze image metadata.

### 2. 🖼️ Skin Lesion Image Preprocessing

Learn how to load skin lesion images, resize them, convert them into tensors, and normalize pixel values for CNN input.

### 3. 🔄 Data Augmentation

Learn how to apply image transformations such as flipping, rotation, and color adjustments to improve training data diversity.

### 4. 🧠 Build a Skin Lesion CNN Classifier

Learn how to build a CNN using convolutional layers, ReLU, pooling, flattening, and fully connected layers to generate seven skin lesion classification scores.

### 5. ⚙️ Transfer Learning (ResNet / EfficientNet)

Learn the fundamentals of transfer learning using pretrained ResNet18, freezing layers, and replacing the classification head for skin lesion categories.

### 6. 📊 Model Evaluation

Learn how to calculate accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices using example classification predictions.

### 7. 🔍 Explainable AI (Grad-CAM)

Learn how to generate Grad-CAM heatmaps to visualize image regions that influence CNN classification scores.

### 8. 📈 Compare Models and Analyze Results

Learn how to compare example classification results for Custom CNN, ResNet18, and EfficientNet-B0 using evaluation metrics and visualizations.

## 🔄 Medical Image Classification Workflow

```text
HAM10000 Dataset
       ↓
Image Preprocessing
       ↓
Data Augmentation (Training Only)
       ↓
CNN Architecture / Pretrained Model
       ↓
Model Training and Validation
       ↓
Model Evaluation
       ↓
Grad-CAM Visualization
       ↓
Model Comparison and Analysis
```

**Note:** This repository focuses on introductory CNN concepts and practical demonstrations. Its current lessons do not train and compare complete HAM10000 classifiers. Model evaluation and comparison examples use simulated predictions, while the seven-class ResNet18 head and Grad-CAM demonstration are untrained for skin lesion diagnosis. The workflow above represents the intended end-to-end process for future experiments.

## 📖 References

- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/)
- [Torchvision Models and Pretrained Weights](https://docs.pytorch.org/vision/stable/models.html)
- [HAM10000 Dataset — Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T)
- [HAM10000 Dataset — Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [ISIC Challenge Datasets](https://challenge.isic-archive.com/data/)
- [PyTorch — Transfer Learning Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Scikit-learn — Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Grad-CAM — PyTorch Implementation](https://github.com/jacobgil/pytorch-grad-cam)

---

**Disclaimer:** This project is for educational and research learning purposes only. It is not a validated medical diagnostic tool.
