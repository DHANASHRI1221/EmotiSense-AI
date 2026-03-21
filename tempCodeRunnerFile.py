from flask import Flask, request, jsonify
import pandas as pd

from preprocess_tfidf import Preprocessor
from src.train import Trainer
from src.predict import Predictor
from src.decision import decide_action
from message import generate_message

app = Flask(__name__)

# ---- Load model ----
train = pd.read_csv("data/train.csv")
train.columns = train.columns.str.strip().str.lower()

text_col = "journal_text"

meta_cols = [
    'sleep_hours','stress_level','energy_level',
    'time_of_day','ambience_type',
    'previous_day_mood','face_emotion_hint',
    'reflection_quality'
]

y_state = train["emotional_state"]
y_intensity = train["intensity"]

pre = Preprocessor()
X_text, X_meta = pre.fit_transform(train, text_col, meta_cols)

trainer = Trainer()
model_state, model_intensity = trainer.train(X_text, X_meta, y_state, y_intensity)

predictor = Predictor(model_state, model_intensity)

# ---- API ----
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    sample = pd.DataFrame([data])

    X_text_s, X_meta_s = pre.transform(sample, text_col, meta_cols)

    pred_s, pred_i, conf, unc = predictor.predict(X_text_s, X_meta_s)

    action, timing = decide_action(
        pred_s[0],
        pred_i[0],
        data["stress_level"],
        data["energy_level"],
        data["time_of_day"],
        data["journal_text"]
    )

    message = generate_message(pred_s[0], pred_i[0], action, conf[0])

    return jsonify({
        "state": pred_s[0],
        "intensity": int(pred_i[0]),
        "confidence": float(conf[0]),
        "action": action,
        "timing": timing,
        "message": message
    })

if __name__ == "__main__":
    app.run(debug=True)