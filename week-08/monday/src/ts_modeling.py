import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_test_split_ts(df, holdout_size=30):
    """
    Split time series into train and test sets respecting temporal order.
    """
    train = df.iloc[:-holdout_size]
    test = df.iloc[-holdout_size:]
    return train, test

def fit_sarima_model(train_series, order=(1,1,1), seasonal_order=(0,0,0,0)):
    """
    Fit a SARIMA model.
    """
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order, 
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    return results

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Calculate evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
