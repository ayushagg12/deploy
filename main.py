from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import os

app = Flask(__name__)

# ==========================================================
# 1. Load Files (Model + Features + Scaler)
# ==========================================================
MODEL_PATH = "model/bilstm_malware_model.keras"   # UPDATED
FEATURES_PATH = "model/feature_names.pkl"
SCALER_PATH = "model/scaler.pkl"

print("📌 Loading model, feature names, and scaler...")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load feature list (list of 639 feature names)
feature_names = joblib.load(FEATURES_PATH)

# Load StandardScaler
scaler = joblib.load(SCALER_PATH)

FEATURE_COUNT = len(feature_names)

print(f"✅ Loaded model!")
print(f"✅ Loaded {FEATURE_COUNT} feature names!")
print(f"✅ Loaded scaler!")

# ==========================================================
# 2. Prediction Function
# ==========================================================
def malware_prediction(features_list):
    try:
        # Convert input to numpy array
        features = np.array(features_list, dtype=np.float32)

        # Validate feature length
        if len(features) != FEATURE_COUNT:
            return {"error": f"Expected {FEATURE_COUNT} features, got {len(features)}."}

        # Validate binary input
        if not np.isin(features, [0, 1]).all():
            return {"error": "Feature values must be only 0 or 1 (binary)."}

        # Scale input (must match training scaler)
        features_scaled = scaler.transform([features])  # shape: (1, 639)

        # Reshape for LSTM input
        features_lstm = features_scaled.reshape(1, 1, FEATURE_COUNT)

        # Predict
        prob = model.predict(features_lstm, verbose=0)[0][0]
        prediction = int(prob > 0.5)

        label = "Benign" if prediction == 1 else "Malware"

        return {
            "prediction": prediction,
            "label": label,
            "probability": float(round(prob, 4)),
            "explanation": "0 = Malware, 1 = Benign"
        }

    except Exception as e:
        return {"error": str(e)}


# ==========================================================
# 3. API Routes
# ==========================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🔥 Malware Detection API (Flask + TensorFlow) Running Successfully!"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if "features" not in data:
            return jsonify({"error": "JSON must include 'features' field"}), 400

        features = data["features"]

        result = malware_prediction(features)

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================================
# 4. Run Flask App Locally
# ==========================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
