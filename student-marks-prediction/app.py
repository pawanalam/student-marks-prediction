"""
Flask application for Student Marks Prediction.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH  = os.path.join("model", "marks_model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

model  = None
scaler = None

def load_artifacts():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return True
    return False

load_artifacts()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not trained yet. Run generate_and_train.py first."}), 503

    try:
        data = request.get_json()
        features = np.array([[
            float(data["study_hours"]),
            float(data["attendance"]),
            float(data["prev_score"]),
            float(data["sleep_hours"]),
            float(data["assignments_done"]),
        ]])

        features_sc = scaler.transform(features)
        prediction  = model.predict(features_sc)[0]
        prediction  = float(np.clip(prediction, 0, 100))

        # Grade logic
        if prediction >= 90:
            grade, color = "A+", "#16a34a"
        elif prediction >= 80:
            grade, color = "A",  "#22c55e"
        elif prediction >= 70:
            grade, color = "B",  "#3b82f6"
        elif prediction >= 60:
            grade, color = "C",  "#f59e0b"
        elif prediction >= 50:
            grade, color = "D",  "#f97316"
        else:
            grade, color = "F",  "#ef4444"

        return jsonify({
            "predicted_marks": round(prediction, 2),
            "grade":  grade,
            "color":  color,
        })

    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
