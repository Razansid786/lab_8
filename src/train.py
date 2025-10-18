import pandas as pd
import numpy as np
import yaml
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def get_model(params):
    """Initialize model based on parameters"""
    model_type = params['model']['type']
    random_state = params['model']['random_state']
    
    if model_type == 'RandomForestRegressor':
        model = RandomForestRegressor(
            n_estimators=params['hyperparameters']['n_estimators'],
            max_depth=params['hyperparameters']['max_depth'],
            min_samples_split=params['hyperparameters']['min_samples_split'],
            min_samples_leaf=params['hyperparameters']['min_samples_leaf'],
            random_state=random_state
        )
    elif model_type == 'GradientBoostingRegressor':
        model = GradientBoostingRegressor(
            n_estimators=params['hyperparameters']['n_estimators'],
            max_depth=params['hyperparameters']['max_depth'],
            random_state=random_state
        )
    elif model_type == 'LinearRegression':
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def train_model():
    """Train the model"""
    print("Starting model training...")
    
    params = load_params()
    
    X = pd.read_csv('data/processed/features.csv')
    y = pd.read_csv('data/processed/target.csv').values.ravel()
    
    test_size = params['data']['test_size']
    random_state = params['data']['random_state']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    model = get_model(params)
    print(f"Training {params['model']['type']}...")
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'train': {
            'rmse': float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            'mae': float(mean_absolute_error(y_train, y_train_pred)),
            'r2': float(r2_score(y_train, y_train_pred))
        },
        'test': {
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            'mae': float(mean_absolute_error(y_test, y_test_pred)),
            'r2': float(r2_score(y_test, y_test_pred))
        }
    }
    
    print("\n=== Training Metrics ===")
    print(f"RMSE: {metrics['train']['rmse']:.2f}")
    print(f"MAE: {metrics['train']['mae']:.2f}")
    print(f"R²: {metrics['train']['r2']:.4f}")
    
    print("\n=== Test Metrics ===")
    print(f"RMSE: {metrics['test']['rmse']:.2f}")
    print(f"MAE: {metrics['test']['mae']:.2f}")
    print(f"R²: {metrics['test']['r2']:.4f}")
    
    os.makedirs('models', exist_ok=True)
    model_path = params['training']['save_model_path']
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    metrics_path = params['training']['metrics_path']
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")
    
    return model, metrics

if __name__ == '__main__':
    train_model()