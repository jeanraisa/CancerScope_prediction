# src/prediction.py

import pandas as pd
import joblib
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.preprocessing import load_scaler
from keras.models import load_model


def load_trained_model(model_path='../models/cancer_prediction_model.h5'):
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at path: {model_path}")
    #try:
        #model = load_model(model_path)
        #return model
    #except Exception as e:
        #raise ValueError(f"Failed to load model. Error: {e}")
    
    # Load the actual model
    model = load_model(model_path)
    return model
    #return load_model(model_path)
    
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
    np.array: Predicted probabilities.
    """
    scaler = load_scaler()
    new_data_scaled = scaler.transform(new_data)
    probability = model.predict(new_data_scaled)[0][0]
    prediction = 1 if probability >= 0.5 else 0
    return prediction
    #return model.predict(new_data_scaled)

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
    prediction = predict(model, example_data)
    print(f"Prediction: {prediction}")