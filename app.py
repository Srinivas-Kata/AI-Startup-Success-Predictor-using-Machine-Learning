import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AI Startup Success Predictor",
    page_icon="🚀",
    layout="centered"
)

st.title("🚀 AI Startup Success Predictor")
st.write("Predict Startup Success using Machine Learning")

# -------------------------------
# Load Model
# -------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "startup_success_predictor.pkl"

if not MODEL_PATH.exists():
    st.error(f"Model file not found:\n{MODEL_PATH}")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("Unable to load model.")
    st.code(str(e))
    st.stop()

# -------------------------------
# Input Form
# -------------------------------
with st.form("prediction_form"):

    funding = st.number_input(
        "Funding Amount",
        min_value=0.0,
        value=10.0
    )

    employees = st.number_input(
        "Employees",
        min_value=1,
        value=50
    )

    experience = st.number_input(
        "Founder Experience",
        min_value=0,
        value=5
    )

    market = st.number_input(
        "Market Size",
        min_value=1.0,
        value=100.0
    )

    stage = st.selectbox(
        "Product Stage",
        ["Idea", "MVP", "Growth", "Scale"]
    )

    revenue = st.selectbox(
        "Has Revenue",
        ["No", "Yes"]
    )

    predict = st.form_submit_button("Predict")

# -------------------------------
# Prediction
# -------------------------------
if predict:

    input_df = pd.DataFrame({
        "funding_millions":[funding],
        "employees":[employees],
        "founders_experience_years":[experience],
        "market_size_millions":[market],
        "product_stage":[stage],
        "has_revenue":[revenue]
    })

    try:
        prediction = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]
        else:
            probability = 0

        if prediction == 1:
            st.success(
                f"Startup is likely to succeed.\n\nProbability: {probability:.2%}"
            )
        else:
            st.error(
                f"Startup has a higher chance of failure.\n\nSuccess Probability: {probability:.2%}"
            )

    except Exception as e:
        st.error("Prediction failed")
        st.code(str(e))
