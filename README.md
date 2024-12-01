# CancerScope Prediction

# Overview
CancerScope Prediction is a machine learning project aimed at predicting whether a the individual has cancer. The project utilizes a dataset containing  medical and lifestyle information for  patients, designed to predict the presence of cancer based on various features various attributes , such as cancer history, age of patiesnt, Smoking, and more, to determine the whether a person has of cancer.

# Dataset
* Age: Integer values representing the patient's age, ranging from 20 to 80.
* Gender: Binary values representing gender, where 0 indicates Male and 1 indicates Female.
* BMI: Continuous values representing Body Mass Index, ranging from 15 to 40.
* Smoking: Binary values indicating smoking status, where 0 means No and 1 means Yes.
* GeneticRisk: Categorical values representing genetic risk levels for cancer, with 0 indicating Low, 1 indicating Medium, and 2 indicating High.
* PhysicalActivity: Continuous values representing the number of hours per week spent on physical activities, ranging from 0 to 10.
* AlcoholIntake: Continuous values representing the number of alcohol units consumed per week, ranging from 0 to 5.
* CancerHistory: Binary values indicating whether the patient has a personal history of cancer, where 0 means No and 1 means Yes.
* Diagnosis: Binary values indicating the cancer diagnosis status, where 0 indicates No Cancer and 1 indicates Cancer.

# Project Structure

CancerScope_prediction/ | | │── README.md │── notebook/ │ └── cancerScope_prediction.ipynb │── src/ │ ├── preprocessing.py │ ├── model.py │ └── prediction.py │── data/ │ ├── train/ │ └── test/ └── models/ ├── scaler.pkl └── cancer_prediction_model.h5
    
# Installation

1. Ensure you have Python 3.7+ installed.

2. Clone this repository:
    `git clone https://github.com/jeanraisa/CancerScope_prediction.git
    cd cancerScope_prediction`

3. Install required libraries:

    `pip install -r requirements.txt`

# Preprocessing Steps
The src/preprocessing.py file contains the following main functions:

load_and_preprocess_data(file_path): Loads the CSV file, splits features and target, scales the features using StandardScaler, and saves the scaler.
load_scaler(): Loads the saved StandardScaler object.

To run preprocessing:

`python src/preprocessing.py`

Model Training
The src/model.py file contains the following main functions:

create_model(): Creates and compiles the  model.

train_model(X_train, y_train, batch_size=32): Trains the model with early stopping.
evaluate_model(model, X_test, y_test): Evaluates the trained model on test data.
plot_training_history(history): Plots the training and validation accuracy/loss.

To train the model:

`python src/model.py`

Model Testing and Prediction
The src/prediction.py file contains the following main functions:

load_trained_model(model_path): Loads the trained model from a file.
predict(model, new_data): Makes predictions using the trained model.

To run predictions:

`python src/prediction.py`

Model Files
Pickle (.pkl) file:
models/scaler.pkl

TensorFlow (.h5) file:

models/cancer_prediction_model.h5


Notebook
The Jupyter notebook notebook/cancerScope_prediction.ipynb provides data analysis, model training, and result visualization. 

To use the notebook:
Open notebook/cancerScope_prediction.ipynb from the Jupyter interface.
