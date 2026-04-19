import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("Setting up paths...")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "enterprise_product_return_dataset_v2.csv")
model_path = os.path.join(BASE_DIR, "models", "return_model_v4_recall_optimized.pkl")


print("Loading dataset...")
df = pd.read_csv(data_path)

print("Dataset loaded successfully.")
print("Original dataset shape:", df.shape)

df = df.drop(columns=["order_id", "customer_id", "order_date"])

print("Dataset shape after dropping ID columns:", df.shape)



for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print("Categorical encoding completed.")



X = df.drop("returned", axis=1)
print("Feature shape:", X.shape)



print("Loading trained model...")
model = joblib.load(model_path)
print("Model loaded successfully.")



print("Sampling data for faster SHAP computation...")
X_sample = X.sample(1000, random_state=42)



print("Creating SHAP explainer...")
explainer = shap.TreeExplainer(model)

print("Calculating SHAP values...")
shap_values = explainer.shap_values(X_sample, check_additivity=False)

print("SHAP values generated successfully.")



if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
    expected_value = explainer.expected_value[1]
else:
    shap_values_class1 = shap_values[:, :, 1]
    expected_value = explainer.expected_value[1]



print("Generating global feature importance plot...")
shap.summary_plot(shap_values_class1, X_sample, plot_type="bar")
plt.show()



print("Generating detailed summary plot...")
shap.summary_plot(shap_values_class1, X_sample)
plt.show()



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