# AI Startup Success Predictor using Machine Learning

## Project Overview
This project predicts whether a startup is likely to be successful using machine learning.

## Workflow
Data Collection → Data Understanding → Data Cleaning → EDA → Feature Engineering → Preprocessing → Model Training → Evaluation → Model Comparison → Deployment.

## Models
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

## Evaluation
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Project Structure

```text
AI_Startup_Success_Predictor_Project/
├── dataset/
│   └── startup_dataset.csv
├── notebook/
│   ├── AI_Startup_Success_Predictor.ipynb
│   └── training.py
├── app/
│   └── app.py
├── models/
│   ├── startup_success_predictor.pkl
│   ├── model_comparison.csv
│   └── feature_importance.csv
├── report/
│   └── AI_Startup_Success_Predictor_Report.pdf
├── presentation/
│   └── AI_Startup_Success_Predictor_Presentation.pptx
├── requirements.txt
└── README.md
```

## Important Note
The supplied coursework requires a dataset of at least 1,000 records and 15 features. The included CSV contains 1,205 records and 21 columns. The included dataset is a synthetic coursework dataset generated for demonstration and model-development purposes; it should not be presented as real-world startup data.
