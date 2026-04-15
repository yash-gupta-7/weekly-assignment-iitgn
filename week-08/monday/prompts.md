# AI Usage Prompts - Week 08 Monday

## Sub-step 1 & 2: Data Loading and Analysis
**Prompt:**
"Create a data generator script that mimics the Olist e-commerce and Pump Sensor datasets. Then write modular functions for stationarity testing using ADF and sensor data cleaning including interpolation and deduplication."

**Critique:**
"The AI provided a good base for data generation and modular cleaning functions. I had to ensure the sensor data interpolation used time-based methodology specifically to handle irregular intervals if they arise."

## Sub-step 3 & 4: Sales Forecasting
**Prompt:**
"Generate code to fit a SARIMA model on the e-commerce sales data. Include a temporal train-test split and evaluation metrics like MAE and MAPE."

**Critique:**
"The implementation correctly used SARIMAX from statsmodels. I added a 30-day hold-out window to reflect a realistic business forecasting cycle."

## Sub-step 5, 6 & 7: Sensor Monitoring
**Prompt:**
"Build a failure prediction model for the next 24 hours using Random Forest. Include a function to calculate business cost based on asymmetric costs (missed failures vs false alarms)."

**Critique:**
"The feature engineering (rolling stats) is appropriate for time-series classification. I refined the cost matrix values to reflect typical industry standards for warehouse equipment repair vs inspection."
