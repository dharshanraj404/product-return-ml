# =====================================
# 📦 Import Required Libraries
# =====================================

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# =====================================
# 📂 Setup Base Directory (Safe Paths)
# =====================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "enterprise_product_return_dataset_v2.csv")
models_dir = os.path.join(BASE_DIR, "models")

os.makedirs(models_dir, exist_ok=True)

print("Loading dataset from:", data_path)

# =====================================
# 📊 Load Dataset
# =====================================

df = pd.read_csv(data_path)
print("Original dataset shape:", df.shape)

# =====================================
# 🔥 Drop ID Columns
# =====================================

df = df.drop(columns=["order_id", "customer_id", "order_date"])
print("Dataset shape after dropping ID columns:", df.shape)

# =====================================
# 🧠 Feature Engineering (V2)
# =====================================

df["return_ratio"] = df["previous_returns"] / (df["customer_order_count_before"] + 1)
df["rating_delay_interaction"] = df["customer_rating"] * df["delivery_delay_days"]
df["festive_discount_risk"] = df["discount_percent"] * df["is_festive_season"]

print("Feature engineering completed.")

# =====================================
# 🎯 Define Features & Target
# =====================================

X = df.drop("returned", axis=1)
y = df["returned"]

# =====================================
# ✂ Train-Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# 🔤 Identify Categorical Columns
# =====================================

categorical_features = [
    "product_category",
    "payment_method",
    "city"
]

# =====================================
# 🏗 Preprocessing Pipeline
# =====================================

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

# ======================================================
# 🌳 RANDOM FOREST MODEL
# ======================================================

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ))
])

print("\nTraining RandomForest V2...")
rf_pipeline.fit(X_train, y_train)

rf_proba = rf_pipeline.predict_proba(X_test)[:, 1]
rf_pred = (rf_proba >= 0.30).astype(int)

print("\n===== RANDOM FOREST V2 =====")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_proba))
print(classification_report(y_test, rf_pred))

# ======================================================
# 🚀 XGBOOST MODEL
# ======================================================

xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=1.5,
        random_state=42,
        eval_metric="logloss"
    ))
])

print("\nTraining XGBoost V2...")
xgb_pipeline.fit(X_train, y_train)

xgb_proba = xgb_pipeline.predict_proba(X_test)[:, 1]
xgb_pred = (xgb_proba >= 0.30).astype(int)

print("\n===== XGBOOST V2 =====")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("ROC-AUC:", roc_auc_score(y_test, xgb_proba))
print(classification_report(y_test, xgb_pred))

# =====================================
# 💾 Save Both Models
# =====================================

rf_model_path = os.path.join(models_dir, "return_model_v2_rf.pkl")
xgb_model_path = os.path.join(models_dir, "return_model_v2_xgb.pkl")

joblib.dump(rf_pipeline, rf_model_path)
joblib.dump(xgb_pipeline, xgb_model_path)

print("\nModels saved:")
print("RandomForest:", rf_model_path)
print("XGBoost:", xgb_model_path)

print("\nTraining & comparison completed successfully!")