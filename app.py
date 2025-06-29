from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS  # Add this import
import numpy as np
import pandas as pd
from tensorflow import keras
import pickle
import json
import os
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model and scaler
try:
    model = keras.models.load_model("stress_detection_model.h5")
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    raise

# Load model metrics and comparison metrics with error handling
try:
    with open("model_metrics.json", "r") as f:
        model_metrics = json.load(f)
except Exception as e:
    logger.warning(f"Error loading model_metrics.json: {str(e)}. Using default values.")
    model_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

try:
    with open("comparison_metrics.json", "r") as f:
        comparison_metrics = json.load(f)
except Exception as e:
    logger.warning(f"Error loading comparison_metrics.json: {str(e)}. Using default values.")
    comparison_metrics = {
        "LSTM": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0},
        "CNN": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0},
        "Transformer": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0},
        "MLP": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
    }

def predict_new_data(model, scaler, new_data):
    if new_data['HR'] < 55 or new_data['HR'] > 140:
        return {"result": "⚠️ Invalid Heart Rate: Must be between 55 and 140 bpm", "confidence": 0.0}
    new_df = pd.DataFrame([new_data])
    new_scaled = scaler.transform(new_df)
    new_reshaped = new_scaled.reshape(1, 1, new_scaled.shape[1])
    predictions = model.predict(new_reshaped)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = float(predictions[0][predicted_class])
    label_map = {
        0: ('Low', '✅ Low stress: Great! Stay stress-free and maintain balance.'),
        1: ('Moderate', '🟡 Moderate stress: Remember to take breaks and relax.'),
        2: ('High', '🔴 High stress: Take deep breaths, meditate, and consider disconnecting for a while.')
    }
    predicted_label, advice = label_map[predicted_class]
    result = f"Predicted Stress Level: {predicted_label}\nConfidence: {confidence:.2%}\nAdvice: {advice}"
    return {"result": result, "confidence": confidence}

@app.route('/')
def index():
    return render_template('index.html', metrics=model_metrics, comparison_metrics=comparison_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        prediction = predict_new_data(model, scaler, data)
        return jsonify(prediction)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'result': f"Error: {str(e)}", 'confidence': 0.0}), 500

@app.route('/static/images/<filename>')
def serve_image(filename):
    return send_from_directory('static/images', filename)

if __name__ == '__main__':
    app.run(debug=False)