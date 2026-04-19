import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

THRESHOLD = 0.30

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(
    BASE_DIR,
    "models",
    "return_model_v2_rf.pkl"
)

print("Loading V2 RandomForest model from:", model_path)


if not os.path.exists(model_path):
    raise FileNotFoundError("V2 model file not found!")

model = joblib.load(model_path)

print("Model loaded successfully!")


app = Flask(__name__)


@app.route("/")
def home():
    return "Product Return Prediction API (V2 Clean Pipeline) Running!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert JSON → DataFrame
        input_df = pd.DataFrame([data])

        #  IMPORTANT: Add engineered features (same as training)
        input_df["return_ratio"] = (
            input_df["previous_returns"] /
            (input_df["customer_order_count_before"] + 1)
        )

        input_df["rating_delay_interaction"] = (
            input_df["customer_rating"] *
            input_df["delivery_delay_days"]
        )

        input_df["festive_discount_risk"] = (
            input_df["discount_percent"] *
            input_df["is_festive_season"]
        )

        # Predict probability
        probability = model.predict_proba(input_df)[:, 1][0]

        # Apply threshold
        prediction = int(probability >= THRESHOLD)

        # Response
        result = {
            "probability_of_return": round(float(probability), 4),
            "threshold_used": THRESHOLD,
            "prediction": prediction,
            "interpretation": (
                "High Return Risk" if prediction == 1
                else "Low Return Risk"
            )
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)