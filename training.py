from IPython.display import display
# AI Startup Success Predictor using Machine Learning

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from xgboost import XGBClassifier

# 1. Load dataset
df = pd.read_csv("../dataset/startup_dataset.csv")
print("Shape:", df.shape)
display(df.head())

# 2. Data understanding
display(df.info())
display(df.describe(include="all").T)
print("Missing values:")
display(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# 3. Data cleaning
df = df.drop_duplicates().reset_index(drop=True)
if "Startup_Name" in df.columns:
    df = df.drop(columns=["Startup_Name"])
df["Startup_Status"] = pd.to_numeric(df["Startup_Status"], errors="coerce")

# 4. EDA
fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(data=df, x="Startup_Status", ax=ax)
ax.set_title("Startup Success Distribution")
plt.show()

fig, ax = plt.subplots(figsize=(9,5))
sns.countplot(data=df, y="Industry", order=df["Industry"].value_counts().index, ax=ax)
ax.set_title("Startup Distribution by Industry")
plt.show()

fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(data=df, x="Funding_Amount", bins=30, kde=True, ax=ax)
ax.set_title("Funding Amount Distribution")
plt.show()

fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(data=df, x="Startup_Status", y="Founder_Experience", ax=ax)
ax.set_title("Founder Experience vs Startup Success")
plt.show()

fig, ax = plt.subplots(figsize=(7,5))
sns.countplot(data=df, x="Competition_Level", hue="Startup_Status", ax=ax)
ax.set_title("Competition Level vs Startup Success")
plt.show()

fig, ax = plt.subplots(figsize=(7,5))
sns.countplot(data=df, x="Market_Size", hue="Startup_Status", ax=ax)
ax.set_title("Market Size vs Startup Success")
plt.show()

numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(14,10))
sns.heatmap(numeric_df.corr(), center=0, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 5. Feature engineering
df["Funding_per_Employee"] = df["Funding_Amount"] / df["Team_Size"].replace(0, np.nan)
df["Founder_Experience_Score"] = df["Founder_Experience"] * df["Number_of_Founders"]
df["Burn_Ratio"] = df["Burn_Rate"] / df["Funding_Amount"].replace(0, np.nan)
df["Customer_Growth_Percentage"] = df["Revenue_Growth"] * 0.7 + df["Customer_Retention"] * 0.3
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 6. Split
X = df.drop(columns=["Startup_Status"])
y = df["Startup_Status"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical_features)
])

# 7. Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=42
    )
}

fitted_models = {}
results = []

for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    fitted_models[name] = pipe

    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob)
    })

results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False).reset_index(drop=True)
display(results_df)

# 8. Reports
for name, model in fitted_models.items():
    pred = model.predict(X_test)
    print("="*70)
    print(name)
    print(classification_report(y_test, pred, target_names=["Failed","Successful"], zero_division=0))

# 9. Confusion matrices
fig, axes = plt.subplots(2,2, figsize=(12,9))
for ax, (name, model) in zip(axes.ravel(), fitted_models.items()):
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["Failed","Successful"], ax=ax
    )
    ax.set_title(name)
plt.tight_layout()
plt.show()

# 10. ROC curves
plt.figure(figsize=(9,6))
for name, model in fitted_models.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name)
plt.title("ROC Curve Comparison")
plt.show()

# 11. Best model
best_model_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_model_name]
print("Best model:", best_model_name)

# 12. Feature importance
model = best_model.named_steps["model"]
prep = best_model.named_steps["preprocessor"]
feature_names = prep.get_feature_names_out()

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
else:
    importances = np.abs(model.coef_[0])

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

display(importance_df.head(15))

plt.figure(figsize=(10,7))
top = importance_df.head(15).sort_values("Importance")
plt.barh(top["Feature"], top["Importance"])
plt.title(f"Top Feature Importances - {best_model_name}")
plt.tight_layout()
plt.show()

# 13. Save model
joblib.dump(best_model, "../models/startup_success_predictor.pkl")
results_df.to_csv("../models/model_comparison.csv", index=False)
importance_df.to_csv("../models/feature_importance.csv", index=False)

print("Model and evaluation files saved.")
