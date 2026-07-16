import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import joblib

def load_preprocessed_arrays(data_dir="data"):
    """Load preprocessed feature and target NumPy arrays."""
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train, random_state=42):
    """Train the Logistic Regression model."""
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Compute and return accuracy, confusion matrix, and classification report."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, cm, report

def save_model(model, filepath="models/logistic_regression_model.joblib"):
    """Serialize the trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved successfully to '{filepath}'")

def main():
    data_dir = "data"
    
    # Load arrays
    print("Loading preprocessed arrays...")
    X_train, X_test, y_train, y_test = load_preprocessed_arrays(data_dir)
    
    # Train
    print("Training Logistic Regression model...")
    model = train_logistic_regression(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    accuracy, cm, report = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    
    # Save
    save_model(model)

if __name__ == "__main__":
    main()
