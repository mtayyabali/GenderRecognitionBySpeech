# 🎙️ Gender Recognition from Speech using Machine Learning

## 📌 Overview

This project focuses on **automatic gender recognition from speech signals** using supervised machine learning techniques. The goal is to analyze audio recordings, extract meaningful acoustic features, and classify the speaker’s gender by comparing the performance of different machine learning models.

The project was developed as part of my **Bachelor’s Thesis in Software Engineering**, combining concepts from **speech processing, data analysis, and machine learning**.

---

## 🎯 Objectives

* Build an end-to-end **speech classification pipeline** from raw audio to prediction
* Extract relevant acoustic features from speech signals
* Compare multiple supervised learning algorithms
* Evaluate models using standard classification metrics

---

## 🧠 Approach

1. **Audio Preprocessing**

   * Input speech signals are normalized and segmented
   * Noise handling and signal preparation applied

2. **Feature Extraction**

   * Extracted **Mel-Frequency Cepstral Coefficients (MFCC)** from audio samples
   * Experimented with different MFCC dimensions to analyze impact on accuracy

3. **Model Training**

   * Implemented and trained multiple classifiers:

     * Support Vector Machine (SVM)
     * Multi-Layer Perceptron (MLP)
     * K-Nearest Neighbors (KNN)

4. **Evaluation**

   * Compared models using:

     * Accuracy
     * Precision
     * Recall
     * F1-score
   * Analyzed how feature size and model choice affect performance

---

## 📊 Results & Insights

* MFCC-based features proved effective for gender classification from speech
* SVM and MLP achieved strong performance across multiple configurations
* Model accuracy varied based on MFCC dimensionality and classifier parameters
* The project demonstrates how **feature engineering and model selection** directly influence classification quality

---

## 🛠️ Technology Stack

* **Language & Runtime**
  * Python 3.x (tested with 3.8+)
* **Core Libraries**
  * NumPy — array and numerical operations
  * SciKit-Learn — classifiers and evaluation (SVM via `sklearn.svm.SVC`, MLP via `sklearn.neural_network.MLPClassifier`)
  * python-speech-features — MFCC extraction (`mfcc`)
  * Joblib — model persistence (`joblib.dump` / `joblib.load`)
  * Wave — reading raw `.wav` audio (`wave` module)
  * Scikit-learn preprocessing — feature scaling (`preprocessing.scale`)
* **Data & Files**
  * WAV audio datasets under `data/` and `pre_data/`
  * Serialized models saved as `.pkl`

---

## 📂 Project Structure (High-Level)

```
├── data/               # Speech datasets
├── feature_extraction/ # MFCC extraction logic
├── models/             # ML model implementations
├── evaluation/         # Metrics and comparison scripts
└── results/            # Experimental results and analysis
```

---

## 🎓 Academic Context

* **Degree:** Bachelor’s in Software Engineering & Management
* **Type:** Bachelor’s Thesis
* **Domain:** Machine Learning, Speech Processing, Data Analysis

---

## 🚀 Future Improvements

* Extend classification to age or emotion recognition
* Experiment with deep learning models (CNNs, RNNs)
* Apply data augmentation to improve generalization
* Deploy as an API or interactive demo