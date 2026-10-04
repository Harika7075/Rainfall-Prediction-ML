"""
Rainfall Prediction using Machine Learning
Internship Project - Zephyr Technologies and Solutions Pvt Ltd

Pipeline (matches project report Section 05 & 06):
1. Data Collection
2. Data Preprocessing (missing values, label encoding, target creation, scaling)
3. Exploratory Data Analysis
4. Feature Selection
5. Model Selection (Logistic Regression, Decision Tree, Random Forest)
6. Model Training (80/20 split)
7. Model Evaluation (Accuracy, Classification Report, Confusion Matrix)
8. Save best model

NOTE: DISTRICT (as a fixed named category) is NOT used as a model
feature. A label encoder for district names only ever recognizes the
exact names it was trained on, so it breaks the moment you ask about
anywhere new. LATITUDE/LONGITUDE are used instead, which generalizes
to any location without retraining. See api/predict_live.py for a
live prediction script that works for any place.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

GRAPH_DIR = "../graphs"
MODEL_DIR = "../models"

# ----------------------------------------------------------------
# 1. DATA COLLECTION
# ----------------------------------------------------------------
df = pd.read_csv("../dataset/rainfall_dataset.csv")
print("Dataset shape:", df.shape)
print(df.head())

# ----------------------------------------------------------------
# 2. DATA PREPROCESSING
# ----------------------------------------------------------------
# 2a. Check missing values
print("\nMissing values:\n", df.isnull().sum())

# 2b. Label Encoding for the categorical PSC / WSC columns
le_psc = LabelEncoder()
le_wsc = LabelEncoder()
df["PSC_ENC"] = le_psc.fit_transform(df["PSC"])
df["WSC_ENC"] = le_wsc.fit_transform(df["WSC"])

# 2c. Create target variable: Rainfall_Class
df["Rainfall_Class"] = (df["PRECTOTCORR"] > 0).astype(int)
print("\nRainfall_Class distribution:\n", df["Rainfall_Class"].value_counts())

# 2d. Features and target
# PRECTOTCORR excluded (target is derived from it -> data leakage).
# DISTRICT (text) and the original PSC/WSC text columns excluded too,
# since their encoded numeric versions (PSC_ENC, WSC_ENC) are used
# instead, and DISTRICT's name itself is dropped in favour of its
# LATITUDE/LONGITUDE, which already exist as columns.
#
# Columns are selected explicitly, in a fixed order (matching
# api/common_features.py's FEATURE_ORDER exactly) rather than relying
# on pandas' column insertion order from .drop() — scikit-learn
# enforces strict feature-name/order matching at predict time, so any
# script using the saved model/scaler MUST build its input this way.
FEATURE_ORDER = [
    "YEAR", "MO", "DY", "RH2M", "T2MDEW", "QV2M", "PS", "WS50M",
    "T2MWET", "WD50M", "T2M_MAX", "T2M_MIN", "ALLSKY_SFC_UV_INDEX", "TS",
    "PSC_ENC", "WSC_ENC", "LATITUDE", "LONGITUDE",
]
X = df[FEATURE_ORDER]
y = df["Rainfall_Class"]

# 2e. Train/test split BEFORE scaling (avoids leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2f. Feature scaling (fit on train, apply to both)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="Rainfall_Class", data=df, hue="Rainfall_Class",
              palette=["#1B2A5E", "#2E75B6"], legend=False)
plt.title("Rainfall Class Count")
plt.xlabel("Rainfall Class (0 = No Rain, 1 = Rain)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/Rainfall_Class_Count.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 4))
sns.histplot(df["PRECTOTCORR"], bins=40, kde=True, color="#1B2A5E")
plt.title("Rainfall Distribution (PRECTOTCORR)")
plt.xlabel("Rainfall (mm)")
plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/Rainfall_Distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df.drop(columns=["DISTRICT", "PSC", "WSC"]).corr(numeric_only=True),
            cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/Correlation_Heatmap.png", dpi=150)
plt.close()

print("\nEDA graphs saved to", GRAPH_DIR)

# ----------------------------------------------------------------
# 5 & 6. MODEL SELECTION + TRAINING
# ----------------------------------------------------------------
results = {}

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
results["Logistic Regression"] = accuracy_score(y_test, y_pred_lr)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42, max_depth=8)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)
results["Decision Tree"] = accuracy_score(y_test, y_pred_dt)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
results["Random Forest"] = accuracy_score(y_test, y_pred_rf)

# ----------------------------------------------------------------
# 7. MODEL EVALUATION
# ----------------------------------------------------------------
print("\n===== MODEL ACCURACY COMPARISON =====")
for name, acc in results.items():
    print(f"{name}: {acc*100:.2f}%")

best_model_name = max(results, key=results.get)
best_pred = {"Logistic Regression": y_pred_lr, "Decision Tree": y_pred_dt, "Random Forest": y_pred_rf}[best_model_name]
best_model = {"Logistic Regression": lr, "Decision Tree": dt, "Random Forest": rf}[best_model_name]

print(f"\nBest model: {best_model_name} ({results[best_model_name]*100:.2f}%)")
print("\nClassification Report (best model):\n",
      classification_report(y_test, best_pred, target_names=["No Rain", "Rain"]))

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/Confusion_Matrix.png", dpi=150)
plt.close()

# Accuracy comparison bar chart
plt.figure(figsize=(6, 4))
plt.bar(results.keys(), [v*100 for v in results.values()], color=["#8FA0C8", "#1B2A5E", "#2E75B6"])
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 100)
for i, v in enumerate(results.values()):
    plt.text(i, v*100 + 1, f"{v*100:.2f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/Model_Accuracy_Comparison.png", dpi=150)
plt.close()

# ----------------------------------------------------------------
# 8. SAVE BEST MODEL
# ----------------------------------------------------------------
with open(f"{MODEL_DIR}/{best_model_name.lower().replace(' ', '_')}_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(f"{MODEL_DIR}/label_encoder_psc.pkl", "wb") as f:
    pickle.dump(le_psc, f)
with open(f"{MODEL_DIR}/label_encoder_wsc.pkl", "wb") as f:
    pickle.dump(le_wsc, f)

print(f"\nSaved best model to {MODEL_DIR}/{best_model_name.lower().replace(' ', '_')}_model.pkl")
print("Project run complete.")
