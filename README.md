# 🧠 EmotiSense AI

### Hybrid Emotion Intelligence & Mental Wellness Assistant

🚀 **EmotiSense AI** is a hybrid machine learning system that analyzes user emotions using text + behavioral signals and provides intelligent, actionable wellness recommendations.

It combines:

* 🔹 **TF-IDF + XGBoost** (pattern-based understanding)
* 🔹 **BERT embeddings + XGBoost** (semantic understanding)
* 🔹 **Hybrid Ensemble Model** (best of both worlds)

---

## 🌐 Live Demo

Click below to watch the full demo:

<p align="center">
  <a href="https://www.loom.com/share/279b6c42ce534a6fbf17090b89293598">
    <img src="images/Thumbnail.png" width="600">
  </a>
</p>

---

# 🧠 How It Works

```text
User Input (Text + Context)
        ↓
Preprocessing (TF-IDF / BERT + Features)
        ↓
Parallel Models
   TF-IDF      BERT
        ↓        ↓
     Hybrid Ensemble
        ↓
Prediction + Confidence
        ↓
Decision Engine (Action)
        ↓
Interactive UI
```

---

# 📸 App Walkthrough

## 🧠 AI Analysis (Model Intelligence)

* TF-IDF captures keyword patterns
* BERT understands deeper context
* Hybrid combines both for better reliability

---

## 📊 Confidence Visualization

Each model provides a confidence score, visualized using clean progress bars.

* Helps users trust predictions
* Highlights model certainty
* Enables better decision-making

---

## 🌿 AI Response & Action

The system generates a personalized response along with a recommended action.

✨ **Examples:**

* Breathing exercises
* Grounding techniques
* Deep work suggestions

---

## 💬 Conversation Interface

A clean chat-style interface improves user interaction and usability.

✨ **Features:**

* Emotion-based color coding
* Confidence indicators
* Smooth conversational flow

---

# ⚙️ Key Features

* 🧠 Hybrid emotion classification system
* 📊 Confidence-based predictions
* 🔍 Model comparison & explainability
* 🎯 Context-aware decision engine
* 🌿 Mental wellness recommendations
* 💬 Interactive chat UI

---

# 🧠 Tech Stack

| Category      | Tools                       |
| ------------- | --------------------------- |
| ML Models     | XGBoost, Scikit-learn       |
| NLP           | SentenceTransformers (BERT) |
| Frontend      | Streamlit                   |
| Backend       | Flask (optional)            |
| Data          | Pandas, NumPy               |
| Model Storage | Joblib                      |

---

# 📊 Model Design

| Model  | Role                           |
| ------ | ------------------------------ |
| TF-IDF | Captures keyword patterns      |
| BERT   | Captures semantic meaning      |
| Hybrid | Improves accuracy & robustness |

---

# 🔍 Error Analysis Insights

* BERT → accurate but low confidence
* TF-IDF → overconfident sometimes
* Hybrid → balanced & more reliable

📄 Detailed analysis: `docs/ERROR_ANALYSIS.md`

---

# 🚀 How to Run Locally

### 🔹 Clone Repository

```bash
git clone https://github.com/your-username/EmotiSense-AI.git
cd EmotiSense-AI
```

### 🔹 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the App

### ✅ Recommended (Standalone)

```bash
streamlit run app_streamlit.py
```

### ⚙️ API + Frontend Mode

```bash
python api.py
streamlit run app.py
```

---

# 📁 Project Structure

```text
├── app.py
├── app_streamlit.py
├── api.py
├── src/
├── models/
├── outputs/
├── docs/
├── README.md
```

---

# 💡 Key Learnings

* Hybrid models outperform individual models
* Confidence calibration is critical in ML systems
* Combining text + context improves predictions
* Explainable UI improves user trust

---

# 🚀 Future Improvements

* 🔬 Fine-tuned transformer model
* 📊 Emotion trend tracking
* 🧠 Explainability (SHAP)
* 🎯 Personalized recommendations

---

# 👨‍💻 Author

**Dhanashri Shivdas**



---

# ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
