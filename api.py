from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

from src.predict import Predictor
from src.decision import decide_action
from message import generate_message

app = Flask(__name__)

# =========================
# 🔹 LOAD MODELS
# =========================
model_state_tfidf = joblib.load("models/state_tfidf.pkl")
model_intensity_tfidf = joblib.load("models/intensity_tfidf.pkl")

model_state_bert = joblib.load("models/state_bert.pkl")
model_intensity_bert = joblib.load("models/intensity_bert.pkl")

tfidf_pre = joblib.load("models/tfidf_pre.pkl")
bert_pre = joblib.load("models/bert_pre.pkl")

le_state = joblib.load("models/le_state.pkl")
le_intensity = joblib.load("models/le_intensity.pkl")

scaler_bert = joblib.load("models/scaler_bert.pkl")

# =========================
# 🔹 INIT PREDICTOR
# =========================
predictor = Predictor(
    model_state_tfidf,
    model_intensity_tfidf,
    model_state_bert,
    model_intensity_bert
)

text_col = "journal_text"

meta_cols = [
    'sleep_hours',
    'stress_level',
    'energy_level',
    'time_of_day',
    'ambience_type',
    'previous_day_mood',
    'face_emotion_hint',
    'reflection_quality'
]

# =========================
# 🔹 HEALTH CHECK
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API is running 🚀"})


# =========================
# 🔹 PREDICT API
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.json
        sample = pd.DataFrame([data])

        # =========================
        # 🔹 PREPROCESS
        # =========================
        X_text_tfidf, X_meta = tfidf_pre.transform(sample, text_col, meta_cols)

        X_text_bert = bert_pre.transform(sample[text_col].tolist())
        X_text_bert = scaler_bert.transform(X_text_bert)

        # =========================
        # 🔹 MODEL PREDICTION
        # =========================
        pred = predictor.predict(
            X_text_tfidf,
            X_text_bert,
            X_meta
        )

        # =========================
        # 🔹 TF-IDF OUTPUT
        # =========================
        state_tfidf = le_state.inverse_transform(pred["tfidf"]["state"])[0]
        intensity_tfidf = int(le_intensity.inverse_transform(pred["tfidf"]["intensity"])[0])
        conf_tfidf = float(pred["tfidf"]["confidence"][0])

        # =========================
        # 🔹 BERT OUTPUT
        # =========================
        state_bert = le_state.inverse_transform(pred["bert"]["state"])[0]
        intensity_bert = int(le_intensity.inverse_transform(pred["bert"]["intensity"])[0])
        conf_bert = float(pred["bert"]["confidence"][0])

        # =========================
        # 🔥 HYBRID ENSEMBLE
        # =========================
        bert_weight = 0.7

        tfidf_probs = pred["tfidf"]["probs"][0]
        bert_probs = pred["bert"]["probs"][0]

        final_probs = bert_weight * bert_probs + (1 - bert_weight) * tfidf_probs

        hybrid_idx = int(np.argmax(final_probs))
        hybrid_conf = float(np.max(final_probs))

        state_hybrid = le_state.inverse_transform([hybrid_idx])[0]
        intensity_hybrid = intensity_bert  # better signal

        # =========================
        # 🔹 DECISION ENGINE
        # =========================
        action_hybrid, timing_hybrid = decide_action(
            state_hybrid,
            intensity_hybrid,
            data["stress_level"],
            data["energy_level"],
            data["time_of_day"],
            data["journal_text"]
        )

        message_hybrid = generate_message(
            state_hybrid,
            intensity_hybrid,
            action_hybrid,
            hybrid_conf,
            "hybrid"
        )

        # =========================
        # 🔹 MODEL AGREEMENT
        # =========================
        agreement = (state_tfidf == state_bert)

        # =========================
        # 🔹 RESPONSE
        # =========================
        return jsonify({
            "hybrid": {
                "state": state_hybrid,
                "intensity": intensity_hybrid,
                "confidence": round(hybrid_conf, 3),
                "action": action_hybrid,
                "timing": timing_hybrid,
                "message": message_hybrid
            },
            "tfidf": {
                "state": state_tfidf,
                "confidence": round(conf_tfidf, 3)
            },
            "bert": {
                "state": state_bert,
                "confidence": round(conf_bert, 3)
            },
            "meta": {
                "models_agree": bool(agreement)
            }
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# =========================
# 🔹 RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)