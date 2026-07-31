import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "models" / "startup_success_predictor.pkl"
model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="AI Startup Success Predictor",
    page_icon="🚀",
    layout="centered"
)

st.title("🚀 AI Startup Success Predictor")
st.write("Predict the likelihood of a startup being successful using Machine Learning.")

st.sidebar.header("Startup Information")

funding = st.sidebar.number_input("Funding Amount", min_value=0.0, value=2500000.0)
founders = st.sidebar.number_input("Number of Founders", min_value=1, max_value=20, value=2)
experience = st.sidebar.number_input("Founder Experience (Years)", min_value=0.0, max_value=50.0, value=8.0)

industry = st.sidebar.selectbox(
    "Industry",
    ["FinTech", "HealthTech", "EdTech", "E-Commerce", "AgriTech", "SaaS", "CleanTech"]
)

team_size = st.sidebar.number_input("Team Size", min_value=1, value=50)
market_size = st.sidebar.selectbox("Market Size", ["Small", "Medium", "Large"])
competition = st.sidebar.selectbox("Competition Level", ["Low", "Medium", "High"])
revenue_growth = st.sidebar.number_input("Revenue Growth (%)", value=35.0)
burn_rate = st.sidebar.number_input("Burn Rate", min_value=0.0, value=50000.0)
customer_rating = st.sidebar.slider("Customer Rating", 1.0, 5.0, 4.2)
customers = st.sidebar.number_input("Number of Customers", min_value=0, value=25000)
retention = st.sidebar.slider("Customer Retention (%)", 0.0, 100.0, 78.0)
patents = st.sidebar.number_input("Patent Count", min_value=0, value=3)
investor = st.sidebar.selectbox("Investor Type", ["Angel", "VC", "Bootstrapped"])
registration_year = st.sidebar.number_input("Registration Year", min_value=1900, max_value=2026, value=2021)

if st.button("Predict Startup Success", type="primary"):
    funding_per_employee = funding / max(team_size, 1)
    founder_experience_score = experience * founders
    burn_ratio = burn_rate / max(funding, 1)
    customer_growth = revenue_growth * 0.7 + retention * 0.3

    input_data = pd.DataFrame([{
        "Funding_Amount": funding,
        "Number_of_Founders": founders,
        "Founder_Experience": experience,
        "Industry": industry,
        "Team_Size": team_size,
        "Market_Size": market_size,
        "Competition_Level": competition,
        "Revenue_Growth": revenue_growth,
        "Burn_Rate": burn_rate,
        "Customer_Rating": customer_rating,
        "Number_of_Customers": customers,
        "Customer_Retention": retention,
        "Patent_Count": patents,
        "Investor_Type": investor,
        "Registration_Year": registration_year,
        "Funding_per_Employee": funding_per_employee,
        "Founder_Experience_Score": founder_experience_score,
        "Burn_Ratio": burn_ratio,
        "Customer_Growth_Percentage": customer_growth
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0, 1]
    confidence = max(probability, 1 - probability)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Startup is predicted to be SUCCESSFUL")
    else:
        st.error("❌ Startup is predicted to be NOT SUCCESSFUL")

    col1, col2 = st.columns(2)
    col1.metric("Probability of Success", f"{probability:.2%}")
    col2.metric("Confidence Score", f"{confidence:.2%}")

    st.progress(float(probability))

st.markdown("---")
st.caption("AI Startup Success Predictor | Machine Learning Coursework Project")
