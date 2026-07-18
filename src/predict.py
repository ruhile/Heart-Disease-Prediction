import joblib
import pandas as pd
import os

# Resolve path relative to this script for robust execution from any directory
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/heart_disease_model.pkl"))

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

model = joblib.load(model_path)

sample = pd.DataFrame([{
    "age": 60,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalch": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 2
}])

prediction = model.predict(sample)

print("Prediction:", prediction)
