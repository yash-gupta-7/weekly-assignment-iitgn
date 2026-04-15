import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from ts_analysis import summarize_patterns, plot_decomposition
from sensor_cleaner import clean_sensor_data
from ts_modeling import train_test_split_ts, fit_sarima_model, evaluate_model
from sensor_modeling import prepare_failure_dataset, train_failure_model, calculate_business_cost
from sklearn.model_selection import train_test_split

def main():
    print("--- STARTING EXECUTION ---")
    
    # 1. E-commerce Sales
    print("\n## 1. E-commerce Sales Analysis")
    sales_df = pd.read_csv('data/ecommerce_sales_ts.csv')
    summary = summarize_patterns(sales_df, 'date', 'sales')
    print(f"Summary: {summary}")
    
    # 2. Sensor Cleaning
    print("\n## 2. Sensor Data Cleaning")
    sensor_df = pd.read_csv('data/sensor_data.csv')
    clean_sensor_df = clean_sensor_data(sensor_df)
    
    # 3. Sales Forecasting
    print("\n## 3. Sales Forecasting")
    sales_series = sales_df.set_index(pd.to_datetime(sales_df['date']))['sales']
    train, test = train_test_split_ts(sales_series, holdout_size=30)
    results = fit_sarima_model(train, order=(1,1,1), seasonal_order=(1,1,1,7))
    predictions = results.forecast(steps=30)
    eval_metrics = evaluate_model(test, predictions, "SARIMA")
    
    # 4. Failure Prediction
    print("\n## 4. Failure Prediction")
    X, y = prepare_failure_dataset(clean_sensor_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    rf_model = train_failure_model(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    cost, metrics = calculate_business_cost(y_test, y_pred)
    print(f"Business Cost: ${cost}")
    print(f"Metrics: {metrics}")
    
    print("\n--- EXECUTION COMPLETE ---")

if __name__ == "__main__":
    main()
