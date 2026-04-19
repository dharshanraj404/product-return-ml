# ==========================================
# explain.py
# SHAP Explainability (Stable Final Version)
# ==========================================

import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("Setting up paths...")

# =====================================
# 1️⃣ Setup Paths
# =====================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "enterprise_product_return_dataset_v2.csv")
model_path = os.path.join(BASE_DIR, "models", "return_model_v4_recall_optimized.pkl")

# =====================================
# 2️⃣ Load Dataset
# =====================================

print("Loading dataset...")
df = pd.read_csv(data_path)

print("Dataset loaded successfully.")
print("Original dataset shape:", df.shape)

# Drop same columns used in train.py
df = df.drop(columns=["order_id", "customer_id", "order_date"])

print("Dataset shape after dropping ID columns:", df.shape)

# =====================================
# 3️⃣ Encode Categorical Columns
# =====================================

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print("Categorical encoding completed.")

# =====================================
# 4️⃣ Define Features
# =====================================

X = df.drop("returned", axis=1)
print("Feature shape:", X.shape)

# =====================================
# 5️⃣ Load Trained Model
# =====================================

print("Loading trained model...")
model = joblib.load(model_path)
print("Model loaded successfully.")

# =====================================
# 6️⃣ Sample Data for Faster SHAP
# =====================================

print("Sampling data for faster SHAP computation...")
X_sample = X.sample(1000, random_state=42)

# =====================================
# 7️⃣ Create SHAP Explainer
# =====================================

print("Creating SHAP explainer...")
explainer = shap.TreeExplainer(model)

print("Calculating SHAP values...")
shap_values = explainer.shap_values(X_sample, check_additivity=False)

print("SHAP values generated successfully.")

# =====================================
# 8️⃣ Handle Binary Classification Format
# =====================================

if isinstance(shap_values, list):
    # Old format → list of class arrays
    shap_values_class1 = shap_values[1]
    expected_value = explainer.expected_value[1]
else:
    # New format → 3D array (samples, features, classes)
    shap_values_class1 = shap_values[:, :, 1]
    expected_value = explainer.expected_value[1]

# =====================================
# 9️⃣ Global Feature Importance (Bar)
# =====================================

print("Generating global feature importance plot...")
shap.summary_plot(shap_values_class1, X_sample, plot_type="bar")
plt.show()

# =====================================
# 🔟 Detailed Summary Plot (Beeswarm)
# =====================================

print("Generating detailed summary plot...")
shap.summary_plot(shap_values_class1, X_sample)
plt.show()

# =====================================
# 1️⃣1️⃣ Single Prediction Explanation (Waterfall)
# =====================================

print("Explaining single prediction...")

explanation = shap.Explanation(
    values=shap_values_class1[0],
    base_values=expected_value,
    data=X_sample.iloc[0],
    feature_names=X_sample.columns
)

shap.plots.waterfall(explanation)
plt.show()

print("Explainability completed successfully.")