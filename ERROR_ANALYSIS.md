# 🔍 Error Analysis – EmotiSense AI

## 🎯 Goal

To understand model weaknesses and improve prediction reliability.

---

## ❗ Observed Issues

### 1. Low Confidence from BERT

* BERT often predicted correct labels but with low confidence
* Cause: probability miscalibration

✅ Fix:

* Applied probability calibration
* Introduced hybrid ensemble instead of direct comparison

---

### 2. TF-IDF Overconfidence

* TF-IDF sometimes gave high confidence for incorrect predictions
* Cause: reliance on keyword frequency

✅ Fix:

* Reduced reliance via weighted ensemble (30%)

---

### 3. Subtle Emotion Misclassification

Example:

* "focused but tired" → sometimes misclassified

Cause:

* Lack of nuanced training data

✅ Fix:

* Improved feature combination (text + metadata)

---

### 4. Model Disagreement

* TF-IDF and BERT often disagreed

✅ Solution:

* Used probability-based ensemble instead of selecting one model

---

## 📊 Key Learnings

* Confidence scores from different models are not directly comparable
* Ensemble methods improve robustness
* Contextual features significantly improve performance

---

## 🚀 Improvements Implemented

* Hybrid ensemble (BERT + TF-IDF)
* Feature fusion (text + behavioral data)
* UI-based model explainability

---

## 📌 Future Work

* Fine-tune transformer model
* Use SHAP for interpretability
* Add temporal user behavior tracking
