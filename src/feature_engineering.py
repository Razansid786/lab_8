import pandas as pd
import numpy as np
import yaml
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def engineer_features():
    """Perform feature engineering"""
    print("Starting feature engineering...")
    
    params = load_params()
    df = pd.read_csv('data/processed/cleaned_data.csv')
    
    # Adjust 'price' to your actual target column name
    target_col = 'price'
    
    if target_col not in df.columns:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Target column '{target_col}' not found")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Save engineered features
    X.to_csv('data/processed/features.csv', index=False)
    y.to_csv('data/processed/target.csv', index=False)
    
    # Save preprocessing objects
    os.makedirs('models/preprocessors', exist_ok=True)
    joblib.dump(scaler, 'models/preprocessors/scaler.pkl')
    joblib.dump(label_encoders, 'models/preprocessors/label_encoders.pkl')
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print("Feature engineering completed!")
    
    return X, y

if __name__ == '__main__':
    engineer_features()