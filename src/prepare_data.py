import pandas as pd
import yaml
import os
import sys

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def prepare_data():
    """Load and prepare the dataset"""
    print("Loading data...")
    
    # Load parameters
    params = load_params()
    
    # Load dataset
    df = pd.read_csv('data/zameen_updated.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Basic data cleaning
    df = df.drop_duplicates()
    df = df.dropna()
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/cleaned_data.csv', index=False)
    
    print(f"Cleaned data shape: {df.shape}")
    print("Data preparation completed!")
    
    return df

if __name__ == '__main__':
    prepare_data()