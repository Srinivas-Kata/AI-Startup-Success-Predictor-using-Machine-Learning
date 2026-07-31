# Deployment Guide

## Local
1. Open terminal in the project root.
2. Install dependencies:
   pip install -r requirements.txt
3. Run the training notebook from the notebook folder.
4. Confirm models/startup_success_predictor.pkl exists.
5. Run:
   streamlit run app/app.py

## GitHub
Create a repository and upload the complete project folder.

## Streamlit Community Cloud
1. Sign in to Streamlit Community Cloud.
2. Connect your GitHub account.
3. Select this repository.
4. Set the main file path to:
   app/app.py
5. Deploy.
6. Copy the public application URL into the report, presentation and submission form.

## Important
Run the notebook before deployment so that the model file is present.
