import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

NUMERIC_FEATURES = [
    'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
]
CATEGORICAL_FEATURES = ['Type']
SELECTED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


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
    
    # Drop IDs - we do not need them for our clustering
    df = df.drop(['UDI', 'Product ID'], axis=1)
    
    # Separate labels for later validation
    failure_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    labels = df[failure_cols].copy()
    data = df.drop(failure_cols, axis=1)
    
    logging.info(f"Features shape: {data.shape} | Labels shape: {labels.shape}")
    return data, labels


def get_selected_features() -> list[str]:
    """Return the feature subset used throughout the project workflow."""
    return SELECTED_FEATURES.copy()


def build_preprocessing_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), CATEGORICAL_FEATURES)
        ])
    return preprocessor
