# Plant Disease Detection and Classification

# 🌿 Plant Disease Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![Model](https://img.shields.io/badge/Model-EfficientNet--B0-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

---

This project leverages **Deep Learning techniques** to classify plant diseases from leaf images. It uses a **pretrained EfficientNet-B0 model** with transfer learning to achieve **~99% classification accuracy** on the PlantVillage dataset.

The project is implemented in **PyTorch** and includes data preprocessing, model training, evaluation, and visualization of results such as confusion matrix and classification report.

---

# 🚀 Key Features

* 📂 **Custom Dataset Loader**
  Handles multi-class image datasets with preprocessing and transformations.

* 🧠 **Transfer Learning (EfficientNet-B0)**
  Uses pretrained weights and fine-tunes deeper layers for high performance.

* 🔒 **Layer Freezing Strategy**
  Freezes early layers and trains deeper layers to improve generalization.

* 📉 **Learning Rate Scheduler**
  Dynamically adjusts learning rate for better convergence.

* 📊 **Performance Metrics**

  * Confusion Matrix
  * Classification Report
  * Accuracy Tracking

* 🎯 **High Accuracy**

  * Achieved **~99% accuracy**

* 🔍 **Inference Support**

  * Predicts plant diseases from unseen images

---

# 📂 Dataset

The project uses the **PlantVillage dataset**, containing images of healthy and diseased plant leaves.

## 📁 Dataset Structure

```
/root_dir
   ├── Class_1
   │       ├── img1.jpg
   │       ├── img2.jpg
   │       └── ...
   ├── Class_2
   │       ├── img1.jpg
   │       ├── img2.jpg
   │       └── ...
   └── ...
```

---

# ⚙️ Requirements

```bash
pip install torch torchvision numpy pillow scikit-learn matplotlib seaborn opencv-python
```

---

# 🧠 Code Structure Overview

## 1️⃣ Dataset Preparation

* Custom dataset class for loading images
* Applies resizing and normalization

## 2️⃣ Model Architecture

* EfficientNet-B0 pretrained model
* Frozen early layers
* Custom classifier with dropout

## 3️⃣ Training

* Loss: CrossEntropyLoss
* Optimizer: AdamW
* Scheduler: StepLR

## 4️⃣ Evaluation

* Classification Report
* Confusion Matrix

---

# 📊 Results

| Metric    | Value |
| --------- | ----- |
| Accuracy  | ~99%  |
| Precision | ~99%  |
| Recall    | ~99%  |
| F1 Score  | ~99%  |

---

# 📊 Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

---

# 🧪 Prediction Example

![Sample Input](images/sample_input.png)

**Output:**

```
Predicted Disease: Tomato Bacterial Spot
Confidence: 99%+
```

---

# ▶️ How to Run

## Train Model

```bash
python main.py
```

## Evaluate

* Generates classification report & confusion matrix

## Predict

* Provide image path → get prediction

---

# ⚠️ Important Note

Dataset is clean and lab-controlled → real-world performance may vary.

---

# 🎯 Conclusion

* High accuracy classification
* Transfer learning improves performance
* Suitable for agricultural applications

---

# 📷 Images Folder

```
/images
   ├── confusion_matrix.png
   ├── sample_input.png
```

---

# 💡 Future Improvements

* Grad-CAM visualization
* Real-world dataset
* Web/mobile deployment

---

# 📄 License

This project is licensed under the MIT License.
