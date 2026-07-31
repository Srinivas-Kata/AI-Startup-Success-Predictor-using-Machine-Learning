import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Startup Success Predictor",
    page_icon="🚀",
    layout="centered"
)

st.title("🚀 AI Startup Success Predictor")
st.write("Predict startup success using Machine Learning.")

# -----------------------------
# Model Path
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "startup_success_predictor.pkl"

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("❌ Model file not found!")
    st.write("Expected location:")
    st.code(str(MODEL_PATH))
    st.info("Run train_model.py first or upload startup_success_predictor.pkl into the models folder.")
    st.stop()

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error("❌ Error loading model")
    st.code(str(e))
    st.stop()

# -----------------------------
# Input Form
# -----------------------------
with st.form("prediction_form"):

    funding = st.number_input(
        "Funding (USD Millions)",
        min_value=0.0,
        value=10.0
    )

    employees = st.number_input(
        "Number of Employees",
        min_value=1,
        value=50
    )

    experience = st.number_input(
        "Founders' Experience (Years)",
        min_value=0,
        value=5
    )

    market = st.number_input(
        "Market Size (USD Millions)",
        min_value=1.0,
        value=100.0
    )

    stage = st.selectbox(
        "Product Stage",
        ["Idea", "MVP", "Growth", "Scale"]
    )

    revenue = st.selectbox(
        "Has Revenue?",
        ["No", "Yes"]
    )

    submit = st.form_submit_button("Predict Success")

# -----------------------------
# Prediction
# -----------------------------
if submit:

    input_data = pd.DataFrame({
        "funding_millions": [funding],
        "employees": [employees],
        "founders_experience_years": [experience],
        "market_size_millions": [market],
        "product_stage": [stage],
        "has_revenue": [revenue]
    })

    st.subheader("Input Data")
    st.write(input_data)

    try:
        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]
        else:
            probability = 0.0

        if prediction == 1:
            st.success(
                f"✅ Startup is likely to succeed.\n\nEstimated Probability: {probability:.2%}"
            )
        else:
            st.warning(
                f"⚠️ Startup has a higher risk of failure.\n\nEstimated Success Probability: {probability:.2%}"
            )

    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))
