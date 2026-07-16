import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import joblib

def load_clean_data(filepath="data/heart_clean_day1.csv"):
    """Load the Day 1 cleaned heart disease dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at {filepath}. Please complete Day 1 steps first.")
    return pd.read_csv(filepath)

def handle_missing_values(df):
    """
    Impute numerical columns with their median (robust to outliers)
    and categorical columns with their mode (most frequent label).
    """
    # Impute numerical columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # Impute categorical columns
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    return df

def encode_categorical_features(df):
    """Encode object (categorical string) columns into integer codes using LabelEncoder."""
    cat_cols = df.select_dtypes(include="object").columns
    le_dict = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
        
    return df, le_dict

def preprocess_pipeline(data_path="data/heart_clean_day1.csv", output_dir="data", test_size=0.2, random_state=42):
    """Run the entire preprocessing, train-test split, and feature scaling pipeline."""
    # 1. Load Data
    df = load_clean_data(data_path)
    
    # 2. Handle missing values
    df = handle_missing_values(df)
    
    # 3. Encode categorical features
    df, encoders = encode_categorical_features(df)
    
    # 4. Separate Features & Target
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # 5. Train-Test Split (with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 6. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames for easier handling and metadata preservation
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # 7. Save outputs to output_dir
    os.makedirs(output_dir, exist_ok=True)
    X_train_scaled_df.to_csv(os.path.join(output_dir, "X_train_scaled.csv"), index=False)
    X_test_scaled_df.to_csv(os.path.join(output_dir, "X_test_scaled.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    # Save the fitted scaler and encoders
    joblib.dump(scaler, os.path.join(output_dir, "scaler_day2.joblib"))
    joblib.dump(encoders, os.path.join(output_dir, "encoders_day2.joblib"))
    
    print("Preprocessing completed!")
    print(f"Scaled Train shape: {X_train_scaled_df.shape}, Scaled Test shape: {X_test_scaled_df.shape}")
    print(f"Preprocessed artifacts saved to '{output_dir}/'")
    
    return X_train_scaled_df, X_test_scaled_df, y_train, y_test

if __name__ == "__main__":
    preprocess_pipeline()
