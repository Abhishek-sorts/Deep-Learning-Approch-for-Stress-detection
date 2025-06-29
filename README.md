# Deep-Learning-Approch-for-Stress-detection

Welcome to the **Stress Detection & Mindfulness Intervention System**, a web-based application that leverages deep learning to classify stress levels (Low, Moderate, High) using physiological data and provides personalized mindfulness recommendations.

## Project Overview
\
This project uses a synthetic dataset samples with features like Heart Rate (HR), EEG Alpha, EEG Beta, and Skin Temperature to predict stress levels. Four deep learning models—LSTM, CNN, Transformer, and MLP—are compared, with LSTM selected as the primary model due to its superior performance (up to 88% accuracy). The system is deployed as a Flask web app, featuring a user-friendly interface to input data, visualize model performance, and receive real-time stress predictions with tailored advice.\

## Features

- **Stress Prediction**: Predicts stress levels (Low, Moderate, High) based on user-input physiological data.\
- **Model Comparison**: Evaluates and compares LSTM, CNN, Transformer, and MLP models with metrics (accuracy, precision, recall, F1-score) and visualizations.\
- **Mindfulness Advice**: Provides actionable advice (e.g., "Take deep breaths" for High stress) based on the predicted stress level.\
- **Visualizations**: Displays confusion matrices, precision-recall curves, learning curves, and an accuracy comparison bar plot.\
- **Web Interface**: Built with Flask, HTML, CSS, and JavaScript for an interactive user experience.\


## Repository Structure
```
StressDetectionApp/
├── static/
│   ├── css/
│   │   └── style.css # CSS for UI styling
│   ├── js/
│   │   └── script.js # JavaScript for form submission and API calls
│   └── images/
│   ├── confusion_matrix.png # Confusion matrix plot
│   ├── precision_recall_curve.png # Precision-recall curve plot
│   ├── learning_curve_lstm.png # Learning curve for LSTM
│   ├── learning_curve_cnn.png # Learning curve for CNN
│   ├── learning_curve_transformer.png# Learning curve for Transformer
│   ├── learning_curve_mlp.png # Learning curve for MLP
│   └── accuracy_comparison.png # Accuracy comparison bar plot
├── templates/
│   └── index.html # Main HTML template for the UI
├── app.py # Flask backend with prediction logic
├── main.py # Main Python script for dataset generation and model training
├── stress_detection_model.h5 # Trained LSTM model
├── scaler.pkl # Scaler for data preprocessing
├── synthetic_health_dataset.csv # Synthetic dataset
├── model_metrics.json # Metrics for the LSTM model
├── comparison_metrics.json # Metrics for all models
├── README.md # This file
```
\

## Installations
\
pip install -r requirements.txt
\
Note: Create a requirements.txt file with the following dependencies:\
flask\
flask-cors\
tensorflow\
numpy\
pandas\
scikit-learn\
matplotlib\
seaborn\
\
Run the Apllication\
python app.py\

a Certain API will be Generated \
ex-Open your browser and visit http://127.0.0.1:5000/ to use the app.\


Usage\
Navigate to the homepage to view model performance metrics and visualizations.
Use the "Predict Your Stress Level" form to input HR (bpm), EEG Alpha (Hz), EEG Beta (Hz), and Skin Temperature (°C).
Click "Predict Stress Level" to receive a prediction with confidence score and mindfulness advice.

Dataset
The synthetic dataset (synthetic_health_dataset.csv) contains 5,000 samples with features:

HR: Heart Rate (bpm)\
EEG_Alpha: Alpha wave frequency (Hz)\
EEG_Beta: Beta wave frequency (Hz)\
Skin_Temp: Skin Temperature (°C)\
Label: Stress level (Low, Moderate, High)\
Generated using normal distributions with realistic ranges (e.g., HR for High stress ~95 bpm).\


Models\
LSTM: Primary model with two LSTM layers, achieving ~88% accuracy.\
CNN: 1D convolutional network for spatial pattern detection.\
Transformer: Attention-based model for long-range dependencies.\
MLP: Feedforward baseline for comparison.\
Results\
LSTM outperformed other models with an accuracy of ~88%, as shown in accuracy_comparison.png.\
Visualizations like confusion_matrix.png and precision_recall_curve.png confirm robust classification.\
Learning curves (learning_curve_*.png) indicate stable training with minimal overfitting.\

Contributing\
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request with your changes.\


Acknowledgments
\
Inspired by research on stress detection using physiological signals (e.g., IEEE papers on LSTM and Transformer models).
Built with TensorFlow, Flask, and other open-source tools.
Contact
For questions or feedback, contact your-Abhishek.kattimani2244@gmail.com
