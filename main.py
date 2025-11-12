from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# ===== Load TensorFlow BiLSTM Model =====
MODEL_PATH = "model/bilstm_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ===== Define prediction function =====
def malware_prediction(features):
    try:
        # Convert features to numpy array
        features = np.array(features, dtype=np.float32)

        # Check input validity
        if len(features) != 639:
            return {"error": f"Feature vector must have exactly 639 elements, got {len(features)}."}
        if not np.isin(features, [0, 1]).all():
            return {"error": "All feature values must be 0 or 1 (binary only)."}

        # Reshape for BiLSTM: (batch=1, timesteps=1, features=639)
        features = features.reshape(1, 1, 639)

        # Predict
        prob = model.predict(features)[0][0]
        prediction = int(prob > 0.5)
        label = "Benign" if prediction == 1 else "Malware"

        return {
            "prediction": prediction,
            "label": label,
            "probability": float(prob),
            "explanation": "0 = Malware, 1 = Benign"
        }
    except Exception as e:
        return {"error": str(e)}

# ===== API Routes =====
@app.route('/')
def home():
    return jsonify({"message": "✅ Malware Detection API (Flask + TensorFlow) is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        if "features" not in data:
            return jsonify({"error": "Missing 'features' in request JSON."}), 400

        features = data["features"]
        result = malware_prediction(features)

        # Handle internal validation errors
        if "error" in result:
            return jsonify(result), 400

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== Run Flask App =====
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
