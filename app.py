import streamlit as st
import numpy as np
import joblib

loaded = joblib.load("lr_model.pkl")
if isinstance(loaded, tuple):
    model, scaler = loaded
else:
    model = loaded
    scaler = None

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict Creativity", "Test Scenarios"])

def get_creativity_state(score):
    if score <= 80:
        return "Low creativity / Rest mode", "Short focus periods"
    elif score <= 150:
        return "Moderate creativity", "1–2 hours of peak work"
    elif score <= 250:
        return "High creativity", "2–4 hours of sustained focus"
    else:
        return "Ultra-creative / Flow state", "3–5 hours of deep work"

if page == "Predict Creativity":
    st.title("💡 Creativity Burst Predictor")
    st.write("Estimate your *creativity burst score* based on daily habits and work factors.")

    hours_coding = st.slider("Hours Coding", 0, 12, 6)
    coffee_intake = st.slider("Coffee Intake (mg)", 0, 500, 200)
    sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
    commits = st.number_input("Commits Made", 0, 100, 10)
    bugs_reported = st.number_input("Bugs Reported", 0, 50, 2)
    ai_usage = st.slider("AI Usage (hours)", 0, 12, 3)
    cognitive_load = st.slider("Cognitive Load (1-10)", 1, 10, 5)
    task_success = st.slider("Task Success (1-10)", 1, 10, 7)

    features = np.array([[hours_coding, coffee_intake, sleep_hours,
                          commits, bugs_reported, ai_usage,
                          cognitive_load, task_success]])
    if features[0,1] == 0:
        features[0,1] = 50

    if scaler:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features

    prediction = model.predict(features_scaled)[0]
    state, duration = get_creativity_state(prediction)

    st.subheader("🎯 Predicted Creative Burst")
    st.metric(label="Creativity Score", value=f"{prediction:.2f}")
    st.write(f"**Current state:** {state}")
    st.write(f"**Expected productive duration:** {duration}")
    st.write(
        "This score depends on a mix of coding hours, rest, caffeine, AI use, and productivity factors."
    )

elif page == "Test Scenarios":
    st.title("🧪 Test Multiple Scenarios")
    st.write("Run predefined example scenarios to see predicted creativity bursts.")

    if st.button("Run Example Scenarios"):
        test_cases = [
            [6, 0, 7, 10, 2, 3, 5, 7],
            [8, 200, 6, 15, 5, 4, 6, 8],
            [3, 50, 8, 5, 1, 1, 3, 5],
            [0, 0, 5, 0, 0, 0, 1, 3],
            [10, 300, 9, 20, 10, 8, 8, 9]
        ]
        labels = [
            "Zero coffee intake, moderate coding and AI usage",
            "High coffee intake, high coding, active developer",
            "Moderate coffee, low-to-moderate activity",
            "Very low activity day, minimal productivity",
            "High productivity day with max inputs"
        ]
        test_array = np.array(test_cases)
        test_array[:,1] = np.where(test_array[:,1] == 0, 50, test_array[:,1])
        if scaler:
            test_array_scaled = scaler.transform(test_array)
        else:
            test_array_scaled = test_array

        preds = model.predict(test_array_scaled)
        for i, pred in enumerate(preds):
            state, duration = get_creativity_state(pred)
            st.write(f"**Scenario {i+1}:** {labels[i]} → Predicted Creativity Burst: **{pred:.2f}**")
            st.write(f"Current state: {state}, Expected productive duration: {duration}\n")
