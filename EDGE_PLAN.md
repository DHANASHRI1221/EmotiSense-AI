# 🚀 Edge Plan – EmotiSense AI

## 🎯 Objective

To build a hybrid emotion intelligence system that combines semantic understanding (BERT) and keyword-based signals (TF-IDF) with contextual metadata for accurate emotional state prediction and actionable insights.

---

## 🧠 System Design Strategy

### 1. Multi-Model Architecture

* TF-IDF + XGBoost → captures keyword patterns
* BERT embeddings + XGBoost → captures semantic meaning
* Hybrid Ensemble → combines both for improved accuracy

---

### 2. Feature Engineering

* Text input (journal entries)
* Behavioral signals:

  * Sleep hours
  * Stress level
  * Energy level
  * Time of day
  * Environment (ambience)
  * Previous mood
  * Reflection quality

---

### 3. Ensemble Decision Logic

* Weighted probability combination:

  * BERT (70%)
  * TF-IDF (30%)
* Final prediction based on combined probabilities

---

### 4. Confidence Calibration

* Used calibrated probabilities for reliable decision-making
* Avoided raw confidence comparison across models

---

### 5. Decision Engine

Maps emotional state + intensity → actionable recommendation:

* Deep work
* Breathing exercises
* Grounding techniques

---

### 6. UI/UX Strategy

* Hybrid-first design (main output)
* Model comparison for transparency
* Emotion-based color coding
* Confidence visualization

---

## 📈 Future Improvements

* Fine-tuned BERT classifier
* SHAP-based explainability
* User history tracking
* Personalized recommendations
* Mobile-friendly UI

---

## 💡 Key Insight

A hybrid system outperforms individual models by combining:

* semantic understanding (BERT)
* statistical patterns (TF-IDF)
* contextual features (metadata)