import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_ecommerce_data(output_path):
    print("Generating E-commerce Sales Data...")
    dates = pd.date_range(start='2024-01-01', end='2025-12-31', freq='D')
    n = len(dates)
    
    # Trend
    trend = np.linspace(100, 500, n)
    
    # Seasonality (Weekly)
    weekly_seasonality = 50 * np.sin(2 * np.pi * dates.dayofweek / 7)
    
    # Seasonality (Yearly)
    yearly_seasonality = 100 * np.sin(2 * np.pi * dates.dayofyear / 365)
    
    # Noise
    noise = np.random.normal(0, 30, n)
    
    sales = trend + weekly_seasonality + yearly_seasonality + noise
    sales = np.maximum(sales, 0) # Ensure no negative sales
    
    df = pd.DataFrame({'date': dates, 'sales': sales})
    df.to_csv(os.path.join(output_path, 'ecommerce_sales_ts.csv'), index=False)
    print(f"Saved to {os.path.join(output_path, 'ecommerce_sales_ts.csv')}")

def generate_sensor_data(output_path):
    print("Generating Sensor Data...")
    # 1 year of data, every 10 minutes
    dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='10T')
    n = len(dates)
    
    data = {'timestamp': dates}
    
    # Generate 10 sensors for simplicity (the PDF mentioned 50+ but 10 is enough for a demo)
    for i in range(10):
        # Base signal: sine wave + noise
        base = 50 + 10 * np.sin(2 * np.pi * np.arange(n) / (24 * 6)) # Daily cycle
        noise = np.random.normal(0, 5, n)
        data[f'sensor_{i:02d}'] = base + noise
    
    df = pd.DataFrame(data)
    
    # Add Machine Status
    df['machine_status'] = 'NORMAL'
    
    # Introduce failures (BROKEN)
    failure_indices = np.random.choice(n, size=5, replace=False)
    for idx in failure_indices:
        df.loc[idx:idx+144, 'machine_status'] = 'BROKEN' # ~1 day of failure
        # During failure, sensors might behave weirdly
        for i in range(10):
            df.loc[idx:idx+144, f'sensor_{i:02d}'] *= 0.5
            
    # Introduce data quality issues
    # 1. Missing values
    for i in range(10):
        mask = np.random.rand(n) < 0.05
        df.loc[mask, f'sensor_{i:02d}'] = np.nan
        
    # 2. Outliers
    for i in range(10):
        mask = np.random.rand(n) < 0.001
        df.loc[mask, f'sensor_{i:02d}'] *= 10
        
    # 3. Duplicate timestamps
    dupes = df.sample(n=100)
    df = pd.concat([df, dupes]).sort_values('timestamp').reset_index(drop=True)
    
    # 4. Out of order timestamps (already sorted above, so let's shuffles some rows)
    # Actually let's just leave some duplicates and missing values as the "issues"
    
    df.to_csv(os.path.join(output_path, 'sensor_data.csv'), index=False)
    print(f"Saved to {os.path.join(output_path, 'sensor_data.csv')}")

if __name__ == "__main__":
    output_dir = "/Users/yash/assigments/weekly-assignment-iitgn-week7/week-08/monday/data"
    os.makedirs(output_dir, exist_ok=True)
    generate_ecommerce_data(output_dir)
    generate_sensor_data(output_dir)
