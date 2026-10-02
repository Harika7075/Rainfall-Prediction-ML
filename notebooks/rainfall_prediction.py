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

GRAPH_DIR = "/home/claude/rainproj/graphs"
MODEL_DIR = "/home/claude/rainproj/models"

# ----------------------------------------------------------------
# 1. DATA COLLECTION
# ----------------------------------------------------------------
df = pd.read_csv("/home/claude/rainproj/dataset/rainfall_dataset.csv")
print("Dataset shape:", df.shape)
print(df.head())

# ----------------------------------------------------------------
# 2. DATA PREPROCESSING
# ----------------------------------------------------------------
# 2a. Check missing values
print("\nMissing values:\n", df.isnull().sum())

# 2b. Label Encoding for DISTRICT
le_district = LabelEncoder()
df["DISTRICT"] = le_district.fit_transform(df["DISTRICT"])

# Also encode the categorical PSC / WSC columns (needed for the model)
le_psc = LabelEncoder()
le_wsc = LabelEncoder()
df["PSC"] = le_psc.fit_transform(df["PSC"])
df["WSC"] = le_wsc.fit_transform(df["WSC"])

# 2c. Create target variable: Rainfall_Class
df["Rainfall_Class"] = (df["PRECTOTCORR"] > 0).astype(int)
print("\nRainfall_Class distribution:\n", df["Rainfall_Class"].value_counts())

# 2d. Features and target (PRECTOTCORR excluded to avoid data leakage)
X = df.drop(["PRECTOTCORR", "Rainfall_Class"], axis=1)
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
sns.countplot(x="Rainfall_Class", data=df, palette=["#1B2A5E", "#2E75B6"])
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
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
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
with open(f"{MODEL_DIR}/label_encoder_district.pkl", "wb") as f:
    pickle.dump(le_district, f)

print(f"\nSaved best model to {MODEL_DIR}/{best_model_name.lower().replace(' ', '_')}_model.pkl")
print("Project run complete.")
