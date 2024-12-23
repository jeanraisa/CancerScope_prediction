# src/preprocessing.py

import os
import pandas as pd
from sklearn.preprocessing import  StandardScaler
from sklearn import preprocessing
import joblib

def load_and_preprocess_data(file_path):
    
    # Load data
    data = pd.read_csv(file_path)
    
    
    # Split features and target
    X = data.drop(["Diagnosis"], axis=1)
    y = data["Diagnosis"]
    
    # Scale features
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_scaled, columns=X.columns)
    print(f"Fitting Scaler: Mean = {scaler.mean_}, Scale = {scaler.scale_}")
    os.makedirs("../models", exist_ok=True)
    joblib.dump(scaler, "../models/scaler.pkl")
    #print(f"Scaler fitted and saved to")
    
    return  X_df, y

def load_scaler():
    scaler_path = '../models/scaler.pkl'
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}.")
    return joblib.load(scaler_path)
    
    #return joblib.load('../models/scaler.pkl')

if __name__ == "__main__":
    # Example usage
    data_path = '../data/train/The_Cancer_data_1500_V2.csv'
    X, y = load_and_preprocess_data(data_path)
    print("Data preprocessed and scaler saved.")
    print("Preprocessing complete. Scaler saved to models/scaler.pkl.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)