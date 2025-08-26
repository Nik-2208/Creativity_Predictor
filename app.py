import streamlit as st
import numpy as np
import joblib

loaded = joblib.load("lr_model.pkl")
if isinstance(loaded, tuple):
    model, scaler = loaded
else:
    model = loaded
    scaler = None

st.title("💡 Creativity Burst Predictor")
st.write("Estimate your *creativity burst score* based on daily habits and work factors.")

hours_coding = st.slider("Hours Coding", 0, 12, 6, help="How many hours you spent coding today.")
coffee_intake = st.slider("Coffee Intake (mg)", 0, 500, 200, help="Approximate caffeine consumed in milligrams.")
sleep_hours = st.slider("Sleep Hours", 0, 12, 7, help="How many hours of sleep you got last night.")
commits = st.number_input("Commits Made", 0, 100, 10, help="How many code commits you pushed today.")
bugs_reported = st.number_input("Bugs Reported", 0, 50, 2, help="Number of bugs/issues identified.")
ai_usage = st.slider("AI Usage (hours)", 0, 12, 3, help="Time spent using AI tools for work or study.")
cognitive_load = st.slider("Cognitive Load (1-10)", 1, 10, 5, help="How mentally demanding your work felt today.")
task_success = st.slider("Task Success (1-10)", 1, 10, 7, help="How successful you felt at completing tasks.")

features = np.array([[hours_coding, coffee_intake, sleep_hours,
                      commits, bugs_reported, ai_usage,
                      cognitive_load, task_success]])

if scaler:
    features = scaler.transform(features)

prediction = model.predict(features)[0]

st.subheader("🎯 Predicted Creative Burst")
st.metric(label="Creativity Score", value=f"{prediction:.2f}")

st.write(
    f"Your predicted creativity burst score is **{prediction:.2f}**. \n\n"
    "Higher scores suggest you're in a strong creative state today. \n\n"
    "This score depends on a mix of coding hours, rest, caffeine, AI use, and productivity factors."
)

st.header("🧪 Test Multiple Scenarios")
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
    test_array[:, 1] = np.where(test_array[:, 1] == 0, 50, test_array[:, 1])
    if scaler:
        test_array_scaled = scaler.transform(test_array)
    else:
        test_array_scaled = test_array

    preds = model.predict(test_array_scaled)
    for i, pred in enumerate(preds):
        st.write(f"**Scenario {i+1}:** {labels[i]} → Predicted Creativity Burst: **{pred:.2f}**")
