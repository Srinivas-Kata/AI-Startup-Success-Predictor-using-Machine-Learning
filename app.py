from pathlib import Path
import streamlit as st
import joblib

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "startup_success_predictor.pkl"

st.write("Looking for model at:")
st.code(str(MODEL_PATH))

if not MODEL_PATH.exists():
    st.error("Model file not found!")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

try:
    model = load_model()
except Exception as e:
    st.error("Model could not be loaded.")
    st.code(str(e))
    st.stop()

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
