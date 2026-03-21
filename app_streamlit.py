import streamlit as st
import pandas as pd
import time
import joblib
import numpy as np

from src.predict import Predictor
from src.decision import decide_action
from message import generate_message

st.set_page_config(page_title="EmotiSense AI", layout="centered")

st.title("🧠 EmotiSense AI")
st.caption("Hybrid Emotion Intelligence & Wellness Assistant")

# =========================
# 🔹 LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    model_state_tfidf = joblib.load("models/state_tfidf.pkl")
    model_intensity_tfidf = joblib.load("models/intensity_tfidf.pkl")

    model_state_bert = joblib.load("models/state_bert.pkl")
    model_intensity_bert = joblib.load("models/intensity_bert.pkl")

    tfidf_pre = joblib.load("models/tfidf_pre.pkl")
    bert_pre = joblib.load("models/bert_pre.pkl")

    le_state = joblib.load("models/le_state.pkl")
    le_intensity = joblib.load("models/le_intensity.pkl")

    scaler_bert = joblib.load("models/scaler_bert.pkl")

    predictor = Predictor(
        model_state_tfidf,
        model_intensity_tfidf,
        model_state_bert,
        model_intensity_bert
    )

    return predictor, tfidf_pre, bert_pre, le_state, le_intensity, scaler_bert


predictor, tfidf_pre, bert_pre, le_state, le_intensity, scaler_bert = load_models()

meta_cols = [
    'sleep_hours', 'stress_level', 'energy_level',
    'time_of_day', 'ambience_type',
    'previous_day_mood', 'face_emotion_hint',
    'reflection_quality'
]

# =========================
# 🔹 MEMORY
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

if st.button("🗑️ Clear Chat"):
    st.session_state.history = []

# =========================
# 🔹 MODE
# =========================
mode = st.radio(
    "Choose Mode",
    ["Hybrid (Recommended)", "TF-IDF Only", "BERT Only"]
)

# =========================
# 🎨 COLORS
# =========================
def emotion_color(state):
    colors = {
        "calm": "#4CAF50",
        "happy": "#2196F3",
        "focused": "#009688",
        "overwhelmed": "#F44336",
        "restless": "#FF9800",
        "sad": "#9C27B0"
    }
    return colors.get(state.lower(), "#607D8B")

# =========================
# 🔹 INPUT
# =========================
text = st.text_input("How are you feeling?")

col1, col2, col3 = st.columns(3)

with col1:
    sleep = st.slider("Sleep Hours", 0, 10, 6)
    duration = st.slider("Activity Duration (min)", 0, 180, 30)

with col2:
    stress = st.slider("Stress Level", 1, 5, 3)
    energy = st.slider("Energy Level", 1, 5, 3)

with col3:
    time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "night"])
    ambience = st.selectbox("Ambience", ["home", "work", "outdoor"])

prev_mood = st.selectbox("Previous Day Mood", ["happy", "neutral", "sad"])
face_hint = st.selectbox("Face Emotion Hint", ["happy", "neutral", "sad"])
reflection = st.selectbox("Reflection Quality", ["low", "medium", "high"])

# =========================
# 🔹 SEND
# =========================
if st.button("Analyze"):

    if len(text.strip()) == 0:
        st.warning("Please enter something")
    else:
        payload = {
            "journal_text": text,
            "sleep_hours": sleep,
            "duration_min": duration,
            "stress_level": stress,
            "energy_level": energy,
            "time_of_day": time_of_day,
            "ambience_type": ambience,
            "previous_day_mood": prev_mood,
            "face_emotion_hint": face_hint,
            "reflection_quality": reflection
        }

        with st.spinner("Analyzing your state..."):
            time.sleep(1)

        sample = pd.DataFrame([payload])

        # TF-IDF
        X_text_tfidf, X_meta = tfidf_pre.transform(sample, "journal_text", meta_cols)

        # BERT
        X_text_bert = bert_pre.transform(sample["journal_text"].tolist())
        X_text_bert = scaler_bert.transform(X_text_bert)

        pred = predictor.predict(X_text_tfidf, X_text_bert, X_meta)

        # Decode
        tfidf = {
            "state": le_state.inverse_transform(pred["tfidf"]["state"])[0],
            "confidence": float(pred["tfidf"]["confidence"][0])
        }

        bert = {
            "state": le_state.inverse_transform(pred["bert"]["state"])[0],
            "confidence": float(pred["bert"]["confidence"][0])
        }

        # HYBRID
        final_probs = 0.7 * pred["bert"]["probs"][0] + 0.3 * pred["tfidf"]["probs"][0]

        idx = np.argmax(final_probs)
        conf = float(np.max(final_probs))

        state_hybrid = le_state.inverse_transform([idx])[0]

        action, timing = decide_action(
            state_hybrid,
            2,
            payload["stress_level"],
            payload["energy_level"],
            payload["time_of_day"],
            payload["journal_text"]
        )

        message = generate_message(state_hybrid, 2, action, conf, "hybrid")

        hybrid = {
            "state": state_hybrid,
            "confidence": conf,
            "message": message,
            "action": action
        }

        agree = (tfidf["state"] == bert["state"])

        # =========================
        # 🔹 SELECT MODEL
        # =========================
        if mode == "TF-IDF Only":
            selected = tfidf
            model_used = "TF-IDF"
        elif mode == "BERT Only":
            selected = bert
            model_used = "BERT"
        else:
            selected = hybrid
            model_used = "HYBRID"

        # =========================
        # 🧠 ANALYSIS
        # =========================
        st.subheader("🧠 AI Analysis")

        if mode == "Hybrid (Recommended)":
            st.info("Using combined intelligence")

        color = emotion_color(selected["state"])

        st.markdown(f"""
        <div style="
            padding:14px;
            border-radius:14px;
            background:white;
            border-left:5px solid {color};
        ">
            <b>{model_used}</b><br>
            {selected["state"]}<br>
            <span style="color:{color};font-weight:bold;">
                {selected["confidence"]:.2f}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # =========================
        # 🌿 RESPONSE
        # =========================
        st.subheader("🌿 AI Response")

        if model_used == "HYBRID":
            message = selected["message"]
            action = selected["action"]
        else:
            action = "reflect"
            message = f"You seem {selected['state']}. Take a moment to reset."

        color = emotion_color(selected["state"])

        st.markdown(f"""
        <div style="
            padding:16px;
            border-radius:16px;
            background:white;
            border-left:6px solid {color};
        ">
            <div style="font-size:18px;font-weight:bold;color:{color};">
                {selected['state'].capitalize()}
            </div>
            <p>{message}</p>
        </div>
        """, unsafe_allow_html=True)

        # Store
        st.session_state.history.append({
            "user": text,
            "bot": f"[{model_used}] {message}",
            "confidence": selected["confidence"],
            "state": selected["state"],
            "action": action
        })

# =========================
# 💬 CHAT
# =========================
st.divider()
st.subheader("💬 Conversation")

for chat in st.session_state.history:

    st.markdown(f"""
    <div style="background:#e3f2fd;padding:10px;border-radius:10px;margin:5px;">
        <b>You:</b> {chat['user']}
    </div>
    """, unsafe_allow_html=True)

    color = emotion_color(chat["state"])

    st.markdown(f"""
    <div style="
        background:#f9f9f9;
        padding:10px;
        border-radius:10px;
        margin:5px;
        border-left:4px solid {color};
    ">
        <b>AI:</b> {chat['bot']}
    </div>
    """, unsafe_allow_html=True)