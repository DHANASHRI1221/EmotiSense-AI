import streamlit as st
import requests
import time

st.set_page_config(page_title="Emotion AI", layout="centered")

st.title("🧠 EmotiSense AI")
st.caption("Hybrid Emotion Intelligence & Mental Wellness Assistant")

# =========================
# 🔹 MEMORY
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# 🔹 CLEAR CHAT
# =========================
if st.button("🗑️ Clear Chat"):
    st.session_state.history = []

# =========================
# 🔹 MODE SELECT
# =========================
mode = st.radio(
    "Choose Mode",
    ["Hybrid (Recommended)", "TF-IDF Only", "BERT Only"]
)

# =========================
# 🎨 EMOTION COLORS
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
if st.button("Send"):

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

        with st.spinner("AI is thinking..."):
            time.sleep(1)
            try:
                res = requests.post("http://127.0.0.1:5000/predict", json=payload)
                if res.status_code != 200:
                    st.error("API error")
                    st.stop()
                data = res.json()
            except:
                st.error("API not running")
                st.stop()

        tfidf = data["tfidf"]
        bert = data["bert"]
        hybrid = data["hybrid"]
        agree = data["meta"]["models_agree"]

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
        # 🧠 AI ANALYSIS
        # =========================
        st.subheader("🧠 AI Analysis")

        if mode == "Hybrid (Recommended)":
            if agree:
                st.success("✅ Models agree")
            else:
                st.warning("⚠️ Models disagree (hybrid used)")

            c1, c2, c3 = st.columns(3)

            def show_card(label, data):
                color = emotion_color(data["state"])
                st.markdown(f"""
                <div style="
                    padding:14px;
                    border-radius:14px;
                    background:white;
                    border-left:5px solid {color};
                    margin-bottom:10px;
                ">
                    <b>{label}</b><br>
                    {data["state"]}<br>
                    <span style="color:{color};font-weight:bold;">
                        {data["confidence"]:.2f}
                    </span>
                </div>
                """, unsafe_allow_html=True)

            with c1: show_card("TF-IDF", tfidf)
            with c2: show_card("BERT", bert)
            with c3: show_card("HYBRID ⭐", hybrid)

        else:
            color = emotion_color(selected["state"])
            st.markdown(f"""
            <div style="
                padding:14px;
                border-radius:14px;
                background:white;
                border-left:5px solid {color};
                margin-bottom:10px;
            ">
                <b>{model_used}</b><br>
                {selected["state"]}<br>
                <span style="color:{color};font-weight:bold;">
                    {selected["confidence"]:.2f}
                </span>
            </div>
            """, unsafe_allow_html=True)

        # =========================
        # 📊 CONFIDENCE BAR
        # =========================
        st.subheader("📊 Confidence")

        def confidence_box(label, value):
            color = "#4CAF50" if value > 0.7 else "#FF9800" if value > 0.4 else "#F44336"

            st.markdown(f"""
            <div style="
                background:white;
                padding:12px;
                border-radius:12px;
                margin-bottom:10px;
            ">
                <b>{label}</b>
                <div style="height:10px;background:#eee;border-radius:10px;margin-top:6px;">
                    <div style="width:{value*100}%;height:10px;background:{color};border-radius:10px;"></div>
                </div>
                <small>{value:.2f}</small>
            </div>
            """, unsafe_allow_html=True)

        if mode == "Hybrid (Recommended)":
            confidence_box("TF-IDF", tfidf["confidence"])
            confidence_box("BERT", bert["confidence"])
            confidence_box("HYBRID", hybrid["confidence"])
        else:
            confidence_box(model_used, selected["confidence"])

        # =========================
        # 🌿 AI RESPONSE
        # =========================
        st.subheader("🌿 AI Response")

        if model_used == "HYBRID":
            message = selected["message"]
            action = selected["action"]
        else:
            action = "reflect"
            message = f"You seem {selected['state']}. Take a moment to reset and observe your feelings."

        if action == "box_breathing":
            message += " Try inhale 4 → hold 4 → exhale 4."
        elif action == "grounding":
            message += " Try 5-4-3 grounding technique."
        elif action == "deep_work":
            message += " Try a 25-min focus session."

        color = emotion_color(selected["state"])

        st.markdown(f"""
        <div style="
            padding:16px;
            border-radius:16px;
            background:white;
            border-left:6px solid {color};
            margin-bottom:12px;
        ">
            <div style="font-size:18px;font-weight:bold;color:{color};">
                {selected['state'].capitalize()}
            </div>
            <p>{message}</p>
            <div style="color:gray;font-size:13px;">
                Confidence: {selected['confidence']:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # =========================
        # 💾 STORE CHAT
        # =========================
        st.session_state.history.append({
            "user": text,
            "bot": f"[{model_used}] {message}",
            "confidence": selected["confidence"],
            "state": selected["state"],
            "action": action
        })

# =========================
# 💬 CHAT UI
# =========================
st.divider()
st.subheader("💬 Conversation")

for chat in st.session_state.history:

    # USER
    st.markdown(f"""
    <div style="
        padding:14px;
        border-radius:14px;
        margin:8px;
        background:#e3f2fd;
        border-left:5px solid #2196F3;
    ">
        <b>🧑 You:</b><br>{chat['user']}
    </div>
    """, unsafe_allow_html=True)

    # AI
    color = emotion_color(chat["state"])

    st.markdown(f"""
    <div style="
        padding:14px;
        border-radius:14px;
        margin:8px;
        background:#f9f9f9;
        border-left:5px solid {color};
    ">
        <b>🤖 AI:</b><br>{chat['bot']}
        <div style="color:gray;font-size:13px;margin-top:6px;">
            {chat['state']} | {chat['action']} | {chat['confidence']:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(float(chat["confidence"]))

    # Colored confidence box
    if chat["confidence"] < 0.4:
        st.markdown("""
        <div style="background:#fff3cd;padding:10px;border-radius:10px;color:#856404;">
            ⚠️ Low confidence prediction
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#d4edda;padding:10px;border-radius:10px;color:#155724;">
            ✅ High confidence prediction
        </div>
        """, unsafe_allow_html=True)