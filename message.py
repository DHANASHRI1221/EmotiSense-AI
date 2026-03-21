def generate_message(state, intensity, action, confidence, model_name=None):

    # 🔹 Optional model tag (useful for debugging / UI)
    model_tag = f"[{model_name.upper()}] " if model_name else ""

    # 🔥 LOW CONFIDENCE
    if confidence < 0.4:
        return f"{model_tag}I'm not completely sure, but it seems like you're feeling {state}. Maybe try {action}."

    # 🔥 NEUTRAL BUT BODY STRESSED
    if state == "neutral" and action in ["box_breathing", "rest"]:
        return f"{model_tag}Even if things seem okay, your body shows stress. Let's slow down — try {action}."

    # 🔥 HIGH NEGATIVE STATES
    if state in ["overwhelmed", "restless"]:
        if intensity >= 4:
            return f"{model_tag}You're feeling strongly overwhelmed. Pause and try {action} immediately."
        return f"{model_tag}It seems you're a bit overwhelmed. Try {action} to stabilize."

    # 🔥 POSITIVE STATES
    if state in ["focused", "calm"]:
        return f"{model_tag}You're in a great state. This is a perfect time for {action}."

    # 🔥 MIXED STATE
    if state == "mixed":
        return f"{model_tag}You're experiencing mixed emotions. A small step like {action} can help bring clarity."

    # 🔥 FALLBACK
    return f"{model_tag}Try {action} to improve your state."