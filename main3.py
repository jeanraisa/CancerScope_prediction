from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import pandas as pd
from io import StringIO
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import os
import io
import joblib
import numpy as np
from src.preprocessing import load_scaler, load_and_preprocess_data
from src.prediction import load_trained_model, predict


# Initialize the app
app = FastAPI()

# Define input data schema for prediction
from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    Age: int = Field(..., description="Patient's age (20 to 80 years)", ge=20, le=80)
    Gender: int = Field(..., description="Gender: 0 for Male, 1 for Female", ge=0, le=1)
    BMI: float = Field(..., description="Body Mass Index (15.0 to 40.0)", ge=15, le=40)
    Smoking: int = Field(..., description="Smoking status: 0 for No, 1 for Yes", ge=0, le=1)
    GeneticRisk: int = Field(
        ..., 
        description="Genetic risk levels: 0 (Low), 1 (Medium), 2 (High)", 
        ge=0, 
        le=2
    )
    PhysicalActivity: float = Field(
        ..., 
        description="Hours of physical activity per week (0.0 to 10.0)", 
        ge=0, 
        le=10
    )
    AlcoholIntake: float = Field(
        ..., 
        description="Alcohol units consumed per week (0.0 to 5.0)", 
        ge=0, 
        le=5
    )
    CancerHistory: int = Field(
        ..., 
        description="Personal cancer history: 0 for No, 1 for Yes", 
        ge=0, 
        le=1
    )


# Load the trained model
#MODEL_PATH = "../models/cancer_prediction_model.h5"
#model = load_trained_model(MODEL_PATH)
# Get the absolute path to the models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cancer_prediction_model.h5")

# Prediction endpoint
@app.post("/predict/")
async def make_prediction(input_data: PredictionInput):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        input_df['PhysicalActivity'] = input_df['PhysicalActivity'].clip(upper=10)
        input_df['AlcoholIntake'] = input_df['AlcoholIntake'].clip(upper=5)
        
        model = load_trained_model()  # Ensure this returns the actual model, not the path
        prediction = predict(model, input_df)
        #prediction, probability = predict(MODEL_PATH, input_df)
        #return {"prediction": prediction, "probability": probability}
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Retrain endpoint
@app.post("/retrain/")
async def retrain_model(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to the server
        upload_path = f"data/{file.filename}"
        with open(upload_path, "wb") as f:
            f.write(await file.read())
        
        # Load and preprocess the data
        X, y = load_and_preprocess_data(upload_path)
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define the model structure
        model = Sequential([
            Dense(64, input_dim=X_train.shape[1], activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        
        # Save the retrained model
        model_path = "../models/cancer_prediction_model.h5"
        if not os.path.exists('../models'):
            os.makedirs('../models')
        model.save(model_path)
        
        # Save the scaler (if it has been modified)
        scaler_path = "../models/scaler.pkl"
        joblib.dump(StandardScaler(), scaler_path)
        
        return {"message": "Model retrained successfully and saved."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Test endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Cancer Prediction API!"}



# CORS Middleware
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000"
]
