import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def prepare_failure_dataset(df, window_size=6, forecast_horizon=144): # 144 = 24h if 10 min steps
    """
    Prepare features and targets for failure prediction.
    Target: Is there a failure in the next 24 hours?
    """
    # Create target: 1 if machine_status is 'BROKEN' in the next forecast_horizon steps
    df['target'] = (df['machine_status'] == 'BROKEN').shift(-forecast_horizon).rolling(window=forecast_horizon).max()
    df['target'] = df['target'].fillna(0).astype(int)
    
    # Simple features: rolling mean/std of sensors
    sensor_cols = [c for c in df.columns if 'sensor' in c]
    features = []
    for col in sensor_cols:
        df[f'{col}_mean'] = df[col].rolling(window=window_size).mean()
        df[f'{col}_std'] = df[col].rolling(window=window_size).std()
        features.extend([f'{col}_mean', f'{col}_std'])
        
    df = df.dropna()
    return df[features], df['target']

def train_failure_model(X_train, y_train):
    """
    Train a Random Forest classifier for failure prediction.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def calculate_business_cost(y_true, y_pred, cost_missed_failure=5000, cost_false_alarm=500):
    """
    Calculate the business cost based on confusion matrix.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fn * cost_missed_failure) + (fp * cost_false_alarm)
    return total_cost, {"Missed Fails": fn, "False Alarms": fp}
