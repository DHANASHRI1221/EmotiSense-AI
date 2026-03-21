def decide_action(state, intensity, stress, energy, time_of_day, text=None):

    # 🔥 1. SHORT TEXT HANDLING
    if text is not None and len(text.split()) <= 2:
        return "light_planning", "later_today"

    # 🔥 2. HIGH INTENSITY NEGATIVE STATES (CRITICAL)
    if state in ["overwhelmed", "restless"] and intensity >= 4:
        return "box_breathing", "now"

    # 🔥 3. MODERATE NEGATIVE STATES
    if state in ["overwhelmed", "restless"] and intensity >= 2:
        return "grounding", "now"

    # 🔥 4. LOW ENERGY CASE
    if energy <= 2:
        if time_of_day == "night":
            return "sleep", "tonight"
        else:
            return "rest", "later_today"

    # 🔥 5. HIGH STRESS OVERRIDE
    if stress >= 4:
        return "box_breathing", "now"

    # 🔥 6. POSITIVE PRODUCTIVE STATES
    if state in ["focused", "calm"] and energy >= 3 and stress <= 3:
        return "deep_work", "within_15_min"

    # 🔥 7. MIXED STATE
    if state == "mixed":
        return "light_planning", "later_today"

    # 🔥 8. NEUTRAL STATE
    if state == "neutral":
        if energy >= 3:
            return "light_planning", "within_15_min"
        else:
            return "rest", "later_today"

    # 🔥 9. FALLBACK
    return "light_planning", "later_today"