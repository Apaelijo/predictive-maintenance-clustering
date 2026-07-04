import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def get_project_root() -> Path:
    """
    Returns the project root directory regardless of where the code is executed.
    """
    return Path(__file__).resolve().parents[2]


def load_and_clean_data(filepath: str = "data/raw/ai4i2020.csv") -> pd.DataFrame:
    # Resolve full path relative to project root
    root = get_project_root()
    full_path = root / filepath

    logging.info(f"Resolved full path: {full_path}")

    df = pd.read_csv(full_path)
    logging.info(f"Loaded dataset with shape: {df.shape}")
    
    # Drop IDs
    df = df.drop(['UDI', 'Product ID'], axis=1)
    
    # Separate labels for later validation
    failure_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    labels = df[failure_cols].copy()
    data = df.drop(failure_cols, axis=1)
    
    logging.info(f"Features shape: {data.shape} | Labels shape: {labels.shape}")
    return data, labels


def build_preprocessing_pipeline():
    numeric_features = [
        'Air temperature [K]', 'Process temperature [K]', 
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ]
    categorical_features = ['Type']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])
    return preprocessor
