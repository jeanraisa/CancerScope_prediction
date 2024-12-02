import pandas as pd
import joblib
import os
import sys
from keras.models import load_model

# Ensure that the current directory is part of the system path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def load_trained_model(model_path='../models/cancer_prediction_model.h5'):
    # Ensure the model path is absolute
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory containing prediction.py
    model_path = os.path.join(base_dir, model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at path: {model_path}")

    return load_model(model_path)

def load_scaler():
    # Ensure the scaler path is absolute
    scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/scaler.pkl")
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at path: {scaler_path}")
    
    # Load the scaler and log its attributes
    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded from: {scaler_path}")
    print(f"Scaler Mean: {getattr(scaler, 'mean_', None)}, Scale: {getattr(scaler, 'scale_', None)}")

    # Check if the scaler is fitted (mean_ and scale_ should not be None)
    if getattr(scaler, 'mean_', None) is None or getattr(scaler, 'scale_', None) is None:
        raise ValueError("Scaler is not fitted. Please check the training data and fit the scaler.")

    return scaler

def predict(model, new_data):
    """
    Make predictions using the trained model.

    Args:
    model (keras.models.Model): Trained model.
    new_data (pd.DataFrame): New data for prediction.

    Returns:
    dict: Contains the prediction and probability.
    """
    try:
        # Load the scaler once
        scaler = load_scaler()

        # Scale the input data
        new_data_scaled = scaler.transform(new_data)

        # Predict probabilities
        probability = float(model.predict(new_data_scaled)[0][0])

        # Convert probability to binary prediction (0 or 1)
        prediction = 1 if probability >= 0.5 else 0

        return {"prediction": prediction, "probability": probability}

    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")

if __name__ == "__main__":
    # Load the trained model
    model = load_trained_model()

    # Example data (replace this with your actual new data)
    example_data = pd.DataFrame({
        'Age': [60],
        'Gender': [0],
        'BMI': [16.085313],
        'Smoking': [1],
        'GeneticRisk': [2],
        'PhysicalActivity': [8.146251],
        'AlcoholIntake': [4.148219],
        'CancerHistory': [0]
    })

    # Predict with the model
    prediction = predict(model, example_data)
    print(f"Prediction: {prediction}")
