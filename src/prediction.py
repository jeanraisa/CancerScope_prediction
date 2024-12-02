# src/prediction.py

import pandas as pd
import joblib
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.preprocessing import load_scaler
from keras.models import load_model


def load_trained_model(model_path='../models/cancer_prediction_model.h5'):
    #model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
    #if not os.path.exists(model_path):
        #raise FileNotFoundError(f"Model file not found at path: {model_path}")
    #try:
        #model = load_model(model_path)
        #return model
    #except Exception as e:
        #raise ValueError(f"Failed to load model. Error: {e}")
    
    # Load the actual model
    #model = load_model(model_path)
    #return model
    #return load_model(model_path)
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory containing prediction.py
    model_path = os.path.join(base_dir, model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at path: {model_path}")

    return load_model(model_path)
    
def load_scaler():
    scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/scaler.pkl")
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at path: {scaler_path}")
    
    return joblib.load(scaler_path)

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
        scaler_path = "../models/scaler.pkl"
        print(f"Loading scaler from: {os.path.abspath(scaler_path)}")
        
        # Load the scaler
        scaler = joblib.load("../models/scaler.pkl")
        print(f"Loaded Scaler: Mean = {getattr(scaler, 'mean_', None)}, Scale = {getattr(scaler, 'scale_', None)}")
        
        #print(f"Scaler loaded successfully: {scaler}")
        #print(f"Scaler mean: {scaler.mean_}")
        #print(f"Scaler scale: {scaler.scale_}")

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
    # Example usage
    model = load_trained_model()
    
    # Create some example data (replace this with your actual new data)
    example_data = pd.DataFrame({
        'Age': [60],
        'Gender': [0],
        'BMI': [16.085313 ],
        'Smoking': [1],
        'GeneticRisk': [2],
        'PhysicalActivity': [8.146251],
        'AlcoholIntake': [4.148219],
        'CancerHistory': [0]
    })
    
    #predictions = predict(model, example_data)
    #print("Prediction (probability of cancer):", predictions[0][0])
    scaler = joblib.load("../models/scaler.pkl")
    print(f"Scaler loaded successfully: {scaler}")
    print(f"Scaler mean: {scaler.mean_}, Scale: {scaler.scale_}")

    prediction = predict(model, example_data)
    print(f"Prediction: {prediction}")