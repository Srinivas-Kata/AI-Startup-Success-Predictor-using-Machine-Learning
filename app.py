import streamlit as st
from pathlib import Path
import pandas as pd
import joblib

st.set_page_config(page_title="AI Startup Success Predictor", page_icon="🚀")
MODEL_PATH = Path(__file__).resolve().parent / "models" / "startup_success_predictor.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("🚀 AI Startup Success Predictor")
st.write("Predict startup success using Machine Learning.")


with st.form("prediction"):
    funding=st.number_input("Funding (USD millions)",0.0,1000.0,10.0)
    employees=st.number_input("Number of Employees",1,100000,50)
    experience=st.number_input("Founders' Experience (years)",0,50,5)
    market=st.number_input("Market Size (USD millions)",1.0,100000.0,100.0)
    stage=st.selectbox("Product Stage",["Idea","MVP","Growth","Scale"])
    revenue=st.selectbox("Has Revenue?",["No","Yes"])
    submit=st.form_submit_button("Predict Success")

if submit:
    x=pd.DataFrame([{"funding_millions":funding,"employees":employees,
        "founders_experience_years":experience,"market_size_millions":market,
        "product_stage":stage,"has_revenue":revenue}])
    p=int(model.predict(x)[0]); prob=float(model.predict_proba(x)[0][1])
    if p: st.success(f"Likely to succeed ✅ — estimated probability: {prob:.1%}")
    else: st.warning(f"Higher risk ⚠️ — estimated success probability: {prob:.1%}")
