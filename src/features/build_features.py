import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Domain-derived features
    df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['power_proxy'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
    df['wear_rate_proxy'] = df['Tool wear [min]'] / (df['Rotational speed [rpm]'] + 1)  # avoid div0
    
    return df

def create_full_pipeline(n_components=6):
    # Full pipeline: features -> scale/encode -> PCA
    from src.data.preprocess import build_preprocessing_pipeline
    
    full_pipeline = Pipeline(steps=[
        ('preprocessor', build_preprocessing_pipeline()),
        ('pca', PCA(n_components=n_components, random_state=42))
    ])
    return full_pipeline