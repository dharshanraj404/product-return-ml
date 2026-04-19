# ==========================================
# threshold_tuning.py
# Recall-Focused Threshold Optimization
# ==========================================

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

# Drop same columns as training
df = df.drop(columns=["order_id", "customer_id", "order_date"])

# Encode categorical columns
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop("returned", axis=1)
y = df["returned"]

# Use same train-test split as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================
# 3️⃣ Load Trained Model
# =====================================

print("Loading trained model...")
model = joblib.load(model_path)
print("Model loaded successfully.")

# =====================================
# 4️⃣ Get Prediction Probabilities
# =====================================

print("Generating prediction probabilities...")
y_probs = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (returned)

# =====================================
# 5️⃣ Test Multiple Thresholds
# =====================================

print("\nThreshold Tuning Results:")
print("-------------------------------------------------")
print("Threshold | Precision | Recall | F1 Score")
print("-------------------------------------------------")

best_threshold = 0.5
best_recall = 0

for threshold in np.arange(0.1, 0.91, 0.05):
    y_pred = (y_probs >= threshold).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{threshold:.2f}      | {precision:.3f}     | {recall:.3f}  | {f1:.3f}")

    # Select threshold with highest recall
    if recall > best_recall:
        best_recall = recall
        best_threshold = threshold

print("-------------------------------------------------")
print(f"\nBest Recall Achieved: {best_recall:.3f}")
print(f"Recommended Threshold for High Recall: {best_threshold:.2f}")