from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# ==========================================================
# 1. Load TensorFlow Model and Feature Files
# ==========================================================
MODEL_PATH = "model.h5"
FEATURES_PATH = "features.pkl"

print("📌 Loading model and feature list...")
model = tf.keras.models.load_model(MODEL_PATH)     # Load model.h5
feature_list = joblib.load(FEATURES_PATH)          # Load features.pkl (list of 639 feature names)
feature_count = len(feature_list)

print(f"✅ Model Loaded!")
print(f"✅ Total Features Expected: {feature_count}")


# ==========================================================
# 2. Prediction Function
# ==========================================================
def malware_prediction(features):
    try:
        # Convert to numpy array
        features = np.array(features, dtype=np.float32)

        # VALIDATIONS
        if len(features) != feature_count:
            return {"error": f"Feature vector must have {feature_count} elements, got {len(features)}."}

        if not np.isin(features, [0, 1]).all():
            return {"error": "All feature values must be 0 or 1 (binary only)."}

        # Reshape for LSTM: (batch=1, timesteps=1, features=N)
        features_lstm = features.reshape(1, 1, feature_count)

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
# 3. API Endpoints
# ==========================================================
@app.route("/")
def home():
    return jsonify({"message": "✅ Malware Detection API (Flask + TensorFlow) Running Successfully!"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if "features" not in data:
            return jsonify({"error": "Missing 'features' in request JSON."}), 400

        features = data["features"]
        result = malware_prediction(features)

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================================
# 4. Run Flask App (LOCAL)
# ==========================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
