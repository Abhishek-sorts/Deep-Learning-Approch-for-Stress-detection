# dataset generator code
import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of samples per label
n_samples_per_label = 5000 // 3

# Define ranges for features based on your sample
def generate_data(label, n):
    if label == 'Low':
        HR = np.random.normal(70, 5, n)
        EEG_Alpha = np.random.normal(9, 1, n)
        EEG_Beta = np.random.normal(15, 2, n)
        Skin_Temp = np.random.normal(36.5, 0.4, n)
    elif label == 'Moderate':
        HR = np.random.normal(85, 5, n)
        EEG_Alpha = np.random.normal(10, 1, n)
        EEG_Beta = np.random.normal(18, 2, n)
        Skin_Temp = np.random.normal(36.8, 0.3, n)
    elif label == 'High':
        HR = np.random.normal(95, 5, n)
        EEG_Alpha = np.random.normal(11, 1, n)
        EEG_Beta = np.random.normal(21, 2, n)
        Skin_Temp = np.random.normal(37.0, 0.3, n)
    
    return pd.DataFrame({
        "HR": HR,
        "EEG_Alpha": EEG_Alpha,
        "EEG_Beta": EEG_Beta,
        "Skin_Temp": Skin_Temp,
        "Label": [label]*n
    })

# Generate data for all three classes
df_low = generate_data('Low', n_samples_per_label)
df_moderate = generate_data('Moderate', n_samples_per_label)
df_high = generate_data('High', 5000 - 2 * n_samples_per_label)  # Ensure total = 5000

# Combine and shuffle the dataset
df = pd.concat([df_low, df_moderate, df_high]).sample(frac=1).reset_index(drop=True)

# Save to CSV
df.to_csv("synthetic_health_dataset.csv", index=False)

print("CSV file 'synthetic_health_dataset.csv' created with 5000 rows.")

# model training code
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, precision_recall_curve, average_precision_score
from tensorflow import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten, MultiHeadAttention, LayerNormalization
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os

# Ensure the static/images directory exists
if not os.path.exists("static/images"):
    os.makedirs("static/images")

# Load Dataset
df = pd.read_csv("synthetic_health_dataset.csv")

# Data Preprocessing
def preprocess_data(df):
    X = df[['HR', 'EEG_Alpha', 'EEG_Beta', 'Skin_Temp']]
    y = df['Label'].map({'Low': 0, 'Moderate': 1, 'High': 2})
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Reshape for sequential models (LSTM, CNN, Transformer): (samples, timesteps, features)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    # Flatten for MLP: (samples, features)
    X_flat = X_scaled
    return X_reshaped, X_flat, y, scaler

X_reshaped, X_flat, y, scaler = preprocess_data(df)

# Train-Test Split
X_train_seq, X_test_seq, X_train_flat, X_test_flat, y_train, y_test = train_test_split(
    X_reshaped, X_flat, y, test_size=0.2, random_state=42
)

# Define Models
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=1, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))  # No MaxPooling since timesteps=1
    model.add(Conv1D(32, kernel_size=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_transformer_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    
    # Project input to higher dimension for attention
    projected = Dense(64)(inputs)  # Now shape becomes [None, 1, 64]
    
    # Multi-head attention
    attention_output = MultiHeadAttention(num_heads=2, key_dim=32)(projected, projected)
    attention_output = Dropout(0.2)(attention_output)
    
    # Residual connection with projected input (not original input)
    attention_output = LayerNormalization(epsilon=1e-6)(projected + attention_output)
    
    # Feed-forward network
    ffn_output = Dense(128, activation='relu')(attention_output)
    ffn_output = Dense(64)(ffn_output)  # Match dimension with attention_output
    ffn_output = Dropout(0.2)(ffn_output)
    
    # Another residual connection
    ffn_output = LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
    
    # Final layers
    flat = Flatten()(ffn_output)
    outputs = Dense(64, activation='relu')(flat)
    outputs = Dense(3, activation='softmax')(outputs)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_mlp_model(input_shape):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize and Train Models
models = {
    "LSTM": build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2])),
    "CNN": build_cnn_model((X_train_seq.shape[1], X_train_seq.shape[2])),
    "Transformer": build_transformer_model((X_train_seq.shape[1], X_train_seq.shape[2])),
    "MLP": build_mlp_model((X_train_flat.shape[1],))
}

# Store metrics for comparison
comparison_metrics = {}

for name, model in models.items():
    print(f"\nTraining {name} model...")
    X_train = X_train_seq if name != "MLP" else X_train_flat
    X_test = X_test_seq if name != "MLP" else X_test_flat
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=64, verbose=1)
    
    # Evaluate Model
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Compute Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # Store Metrics
    comparison_metrics[name] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }
    
    # Plot Learning Curve for each model
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f"{name} Model Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"static/images/learning_curve_{name.lower()}.png")
    plt.close()

# Save the best model (LSTM) and scaler
best_model = models["LSTM"]
best_model.save("stress_detection_model.h5")
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Best model (LSTM) saved as 'stress_detection_model.h5' and scaler saved as 'scaler.pkl'")

# Save metrics for LSTM
metrics = comparison_metrics["LSTM"]
with open("model_metrics.json", "w") as f:
    json.dump(metrics, f)
print(f"LSTM Test Metrics: {metrics}")

# Save comparison metrics
with open("comparison_metrics.json", "w") as f:
    json.dump(comparison_metrics, f)
print("Comparison Metrics:", comparison_metrics)

# Plot Accuracy Comparison
plt.figure(figsize=(8, 5))
accuracies = [metrics["accuracy"] for metrics in comparison_metrics.values()]
model_names = list(comparison_metrics.keys())
sns.barplot(x=accuracies, y=model_names, palette='Blues_d')
plt.title("Accuracy Comparison Across Models")
plt.xlabel("Accuracy")
plt.ylabel("Model")
plt.savefig("static/images/accuracy_comparison.png")
plt.close()

# Plot Confusion Matrix for the best model (LSTM)
y_pred_probs = models["LSTM"].predict(X_test_seq)
y_pred = np.argmax(y_pred_probs, axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Moderate', 'High'], yticklabels=['Low', 'Moderate', 'High'])
plt.title("Confusion Matrix (LSTM)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("static/images/confusion_matrix.png")
plt.close()

# Plot Precision-Recall Curve for the best model (LSTM)
plt.figure(figsize=(6, 4))
for i in range(3):
    precision_i, recall_i, _ = precision_recall_curve(y_test == i, y_pred_probs[:, i])
    ap = average_precision_score(y_test == i, y_pred_probs[:, i])
    plt.plot(recall_i, precision_i, label=f"Class {['Low', 'Moderate', 'High'][i]} (AP = {ap:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (LSTM)")
plt.legend()
plt.grid(True)
plt.savefig("static/images/precision_recall_curve.png")
plt.close()

# prediction code
import numpy as np
import pandas as pd
from tensorflow import keras
import pickle

# Load the trained model and scaler
model = keras.models.load_model("stress_detection_model.h5")
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Function to predict stress for new data
def predict_new_data(model, scaler, new_data):
    """
    Predict stress level for new data with HR validation, confidence scores, and health advice
    """
    # Validate Heart Rate
    if new_data['HR'] < 55 or new_data['HR'] > 140:
        return {"result": "⚠️ Invalid Heart Rate: Must be between 55 and 140 bpm", "confidence": 0.0}

    # Convert new data to DataFrame
    new_df = pd.DataFrame([new_data])
    
    # Scale the features
    new_scaled = scaler.transform(new_df)
    
    # Reshape for LSTM
    new_reshaped = new_scaled.reshape(1, 1, new_scaled.shape[1])
    
    # Predict probabilities
    predictions = model.predict(new_reshaped)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = float(predictions[0][predicted_class])
    
    # Map numeric label to stress level and give advice
    label_map = {
        0: ('Low', '✅ Low stress: Great! Stay stress-free and maintain balance.'),
        1: ('Moderate', '🟡 Moderate stress: Remember to take breaks and relax.'),
        2: ('High', '🔴 High stress: Take deep breaths, meditate, and consider disconnecting for a while.')
    }
    predicted_label, advice = label_map[predicted_class]

    result = f"Predicted Stress Level: {predicted_label}\nConfidence: {confidence:.2%}\nAdvice: {advice}"
    return {"result": result, "confidence": confidence}

# Example usage (for testing)
if __name__ == "__main__":
    example_inputs = [
        {'HR': 70, 'EEG_Alpha': 9.5, 'EEG_Beta': 15.5, 'Skin_Temp': 36.5},
        {'HR': 97.0, 'EEG_Alpha': 9.88, 'EEG_Beta': 21.77, 'Skin_Temp': 36.98},
        {'HR': 65, 'EEG_Alpha': 8.902, 'EEG_Beta': 17.562, 'Skin_Temp': 37.514}
    ]
    print("Stress Level Predictions:")
    for i, input_data in enumerate(example_inputs, 1):
        prediction = predict_new_data(model, scaler, input_data)
        print(f"Sample {i}:")
        print(f"Input: {input_data}")
        print(prediction["result"])
        print()