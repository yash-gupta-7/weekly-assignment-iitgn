# Week 08 Monday Daily Assignment - Time Series Analysis & Forecasting
## Git_Link : https://github.com/yash-gupta-7/weekly-assignment-iitgn/tree/main/week-08/monday 

## Project Overview
This project addresses two core business challenges for an e-commerce platform:
1. **Sales Forecasting**: Predicting daily sales for inventory planning using historical transaction data.
2. **Equipment Monitoring**: Predicting industrial sensor failures in warehouse equipment to minimize downtime and repair costs.

## Repository Structure
- `notebooks/`: Contains the main Jupyter Notebook with step-by-step analysis.
- `src/`: Modular Python scripts for analysis, cleaning, and modeling.
  - `ts_analysis.py`: Stationarity testing and decomposition.
  - `sensor_cleaner.py`: Data cleaning utilities for sensor data.
  - `ts_modeling.py`: SARIMA forecasting and evaluation.
  - `sensor_modeling.py`: Failure prediction and cost analysis.
- `data/`: Datasets used (generated synthetically based on Olist and Pump Sensor datasets).

## Technologies Used
- Python 3.9+
- Pandas, NumPy
- Matplotlib, Seaborn
- Statsmodels (SARIMAX)
- Scikit-learn (RandomForest)

## How to Run
1. Navigate to the `week-08/monday/` directory.
2. Ensure dependencies are installed: `pip install pandas numpy matplotlib statsmodels scikit-learn`.
3. Generate data if not present: `python src/data_generator.py`.
4. Open the notebook in `notebooks/week08_monday_assignment.ipynb` and run all cells.

## Results Summary
- **Sales Model**: A SARIMA(1,1,1)x(1,1,1,7) model captures the weekly seasonality and long-term trend of the e-commerce data.
- **Sensor Model**: The Random Forest classifier predicts failures with a focus on minimizing missed failures (False Negatives) due to the high cost of emergency repairs vs. routine inspections.

---
## 🔗 Git Resource
- **Project Repository**: [GitHub Link](https://github.com/yash-gupta-7/weekly-assignment-iitgn)

