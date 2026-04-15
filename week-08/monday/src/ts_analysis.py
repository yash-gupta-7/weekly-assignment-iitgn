import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

def check_stationarity(series, title="Time Series"):
    """
    Perform Augmented Dickey-Fuller test.
    """
    print(f"\n--- Stationarity Test: {title} ---")
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    
    if result[1] <= 0.05:
        print("Conclusion: The series is stationary (reject H0).")
        return True
    else:
        print("Conclusion: The series is non-stationary (fail to reject H0).")
        return False

def plot_decomposition(series, model='additive', period=None):
    """
    Decompose the time series into trend, seasonal, and residual components.
    """
    decomposition = seasonal_decompose(series.dropna(), model=model, period=period)
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    return fig

def summarize_patterns(df, date_col, value_col):
    """
    Identify potential patterns like trend and seasonality.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    
    summary = {
        "mean": df[value_col].mean(),
        "std": df[value_col].std(),
        "is_stationary": check_stationarity(df[value_col]),
        "has_missing": df[value_col].isnull().any()
    }
    return summary
