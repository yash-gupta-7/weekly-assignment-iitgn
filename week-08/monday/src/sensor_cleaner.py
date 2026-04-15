import pandas as pd
import numpy as np

def clean_sensor_data(df, timestamp_col='timestamp'):
    """
    Clean sensor data: handle duplicates, sort, and interpolate missing values.
    """
    print("\n--- Cleaning Sensor Data ---")
    initial_shape = df.shape
    
    # 1. Convert to datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # 2. Handle Duplicates
    df = df.drop_duplicates(subset=[timestamp_col])
    print(f"Removed {initial_shape[0] - df.shape[0]} duplicate timestamps.")
    
    # 3. Sort by timestamp
    df = df.sort_values(timestamp_col)
    
    # 4. Handle Missing Values
    # Identify numeric columns for interpolation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing_before = df[numeric_cols].isnull().sum().sum()
    
    # Interpolate missing values (time-based if index is time)
    df = df.set_index(timestamp_col)
    df[numeric_cols] = df[numeric_cols].interpolate(method='time')
    
    # Fill remaining (at edges) with backfill/forward fill
    df[numeric_cols] = df[numeric_cols].fillna(method='bfill').fillna(method='ffill')
    
    missing_after = df.isnull().sum().sum()
    print(f"Interpolated {missing_before} missing values. Missing values remaining: {missing_after}")
    
    # Reset index for further use
    df = df.reset_index()
    
    return df

def detect_outliers_zscore(df, cols, threshold=3):
    """
    Detect outliers using Z-score and replace with NaN for cleaning.
    """
    for col in cols:
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        outliers = np.abs(z_scores) > threshold
        print(f"Found {outliers.sum()} outliers in {col}.")
        df.loc[outliers, col] = np.nan
    return df
